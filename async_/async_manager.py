
import asyncio
from threading import Thread, RLock
import contextlib
import weakref
import uuid
import functools
import enum
import time

from helpers.util import singleton, urlsafe_uuid, DEBUG, INFO, WARN, ERROR

# shortcut: from async_ import AsyncLoop, AsyncManager, amend_blank_cbs
__all__ = [
    "AsyncLoop",
    "AsyncManager",
    "amend_blank_cbs",
]

# --- NOT USED ---
# class AsyncResult(asyncio.Future):
#     def __init__(self, *args, **kwargs):
#         super(self.__class__, self).__init__(*args, **kwargs)
#
# class AsyncTask(asyncio.Task):
#     """
#     Implementation by using asyncio.
#     NOTE: already had `_all_tasks`/'all_tasks()`, `_current_tasks`/`current_task(loop)`
#      `cancel()` in asyncio.Task,
#      and already had `result()'/`set_result()`, `exception()`/`set_exception()`,
#      `add_done_callback()`/`remove_done_callback()`, `cancel()`/`cancelled()` in base class asyncio.Future.
#     IMPROVE: implements add_progress_callback(), distinguish succeeded/failed(=resolve/reject)?,
#      and pause()/unpause()? by defining a Task Factory(=loop.set_task_factory)
#     """
#     _current_id = 0
#     @classmethod
#     def _incre_id(cls):
#         cls._current_id += 1
#         return cls._current_id
#
#     def __init__(self, *args, **kwargs):
#         super(self.__class__, self).__init__(*args, **kwargs)
#         self.id = self.__class__._incre_id()

class AsyncLoop(enum.Enum):
    Main = 'main'
    UIThread = 'ui_thread'
    WebApp = 'web_app'
    DataProcess = 'data_process'

    def __str__(self):
        return self.value

@singleton
class AsyncManager:
    __mutex__ = RLock()  # re-entrant thread Lock
    __instance__ = None

    def __init__(self):
        cls = self.__class__
        with cls.__mutex__:
            if cls.__instance__ is not None:
                return
            self.all_loops = weakref.WeakValueDictionary()
            main_loop = asyncio.get_event_loop()
            setattr(main_loop, 'id', AsyncLoop.Main)
            self.current_loop = self.all_loops[AsyncLoop.Main] = main_loop
            # if not main_loop.is_running():
            #     main_loop.run_forever()
            self.all_tasks = weakref.WeakValueDictionary()  # define our own task dict
            cls.__instance__ = self
        # class methods depends on cls.__instance__ called from here.
        self.append_new_loop(id=AsyncLoop.UIThread)
        self.append_new_loop(id=AsyncLoop.WebApp)
        self.append_new_loop(id=AsyncLoop.DataProcess)

    @classmethod
    def __ensure_init__(cls):
        with cls.__mutex__:
            cls.__instance__ = cls() if cls.__instance__ is None else cls.__instance__

    @classmethod
    def append_new_loop(cls, id=None):
        with cls.__mutex__:
            def thread_for_asynctask_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()
            loop = asyncio.new_event_loop()
            loop_thread = Thread(target=thread_for_asynctask_loop, args=(loop,))
            loop_thread.setDaemon(True)  # NOTE: key step to keep life span with the main thread
            setattr(loop, 'thread', loop_thread)
            id = id or cls.new_id()
            setattr(loop, 'id', id)

            cls.__ensure_init__()
            cls.__instance__.current_loop = cls.__instance__.all_loops[id] = loop

            loop_thread.start()
            return loop

    @classmethod
    def current_loop(cls):
        """
        :return: an asyncio.loop object which has extra attributes `id` and `thread`.
        """
        cls.__ensure_init__()
        with cls.__mutex__:
            return cls.__instance__.current_loop

    @classmethod
    def get_loop(cls, id) -> asyncio.AbstractEventLoop:
        cls.__ensure_init__()
        with cls.__mutex__:
            try:
                loop = cls.__instance__.all_loops[id]
            except KeyError:
                loop = None
            return loop

    @staticmethod
    def new_id(prefix=None):
        """
        :param prefix: for a task_id usually head with `loop.id`, while `loop` can get from `task.loop`
        :return: an urlsafe_uuid. if prefix is given, will be headed with "{prefix}#".
        """
        return f'{prefix}_{urlsafe_uuid()}' if prefix is not None else f'{urlsafe_uuid()}'

    @staticmethod
    def hack_task(task, id, coro, loop, progress=0.0):
        """
        append extra attrs. NOTE: Python 3.8 implemented task.set_name, .get_coro etc.
        """
        setattr(task, 'id', id)
        setattr(task, 'coro', coro)
        setattr(task, 'loop', loop)
        setattr(task, 'progress', progress)

        def inner_done_cb(fut):  # TODO: fut == task? make sure
            DEBUG(f'[inner_done_callback] task={id}, result={fut.result()}, except={fut.exception()}')
        task.add_done_callback(inner_done_cb)
        return task

    @classmethod
    def create_task(cls, coroutine, given_id=None, new_thread=False, loop: asyncio.AbstractEventLoop = None):
        """
        :param coroutine:
        :param given_id: task_id will be generally retrieved after task creation. if want to
          use task_id in coro, however, caller may get `new_id()` and send it to `create_task()`.
        :param new_thread: asynctask will always be created (and run) in a new thread,
          but if you want to spawn another one, set `True` here.
        :param loop: you may choose to create (and run) an asynctask in an existing loop.
        :return: an asyncio.Task, which has extra attributes `id`, `coro` and `loop`.
        """
        assert asyncio.iscoroutine(coroutine), f'only accept coroutine, but get {type(coroutine)}'
        cls.__ensure_init__()
        with cls.__mutex__:
            if loop is None:
                loop = cls.__instance__.current_loop if not new_thread else cls.append_new_loop()
            # future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            # IMPROVE: if can hack task_id into the coro (like functools.partial()), `given_id` can be deprecated.
            task_id = cls.new_id(prefix=getattr(loop, 'id', None)) if given_id is None else given_id
            task = loop.create_task(coroutine)
            cls.__instance__.all_tasks[task_id] = task
            cls.hack_task(task, task_id, coroutine, loop)
            return task

    @classmethod
    def gather_task(cls, *task_or_coros, given_id=None, new_thread=False, loop: asyncio.AbstractEventLoop = None):
        """
        :param task_or_coros:
        :param given_id: task_id will be generally retrieved after task creation. if want to
          use task_id in coro, however, caller may get `new_id()` and send it to `create_task()`.
        :param new_thread: asynctask will always be created (and run) in a new thread,
          but if you want to spawn another one, set `True` here.
        :param loop: you may choose to create (and run) an asynctask in an existing loop.
        :return: an asyncio.Future, which has extra attributes `id`, `coro` and `loop`.
        """
        cls.__ensure_init__()
        with cls.__mutex__:
            if loop is None:
                loop = cls.__instance__.current_loop if not new_thread else cls.append_new_loop()
            last_type = None
            for task_or_coro in task_or_coros:
                last_type = last_type or type(task_or_coro)
                if last_type != type(task_or_coro):
                    raise TypeError('All items in task_or_coros must have same type.')
            if last_type is asyncio.Task:
                last_task_loop = None
                for task in task_or_coros:
                    last_task_loop = last_task_loop or getattr(task, 'loop', None)
                    if last_task_loop is None or last_task_loop != getattr(task, 'loop', None):
                        raise ValueError('All tasks must have same loop.')
                if loop != last_task_loop:
                    loop = last_task_loop
                    WARN("Given loop is not same with loop of the tasks and is ignored.")
            task_id = cls.new_id(prefix=getattr(loop, 'id', None)) if given_id is None else given_id

            # wrap future as a task. ensure_future() cannot do this.
            async def coro_wait_future(fut):
                await fut
            coroutine = coro_wait_future(asyncio.gather(*task_or_coros, loop=loop))
            task = cls.create_task(coroutine, given_id=task_id, new_thread=False, loop=loop)
            cls.__instance__.all_tasks[task_id] = task
            cls.hack_task(task, task_id, coroutine, loop)
            return task

    @classmethod
    def run_task(cls, task_or_coro, new_thread=False, loop=None):
        """
        :param task_or_coro: a coro will be wrapped to a task (in new thread or given loop)
           and then run. a task will be run directly, other args will be ignored_.
        :param new_thread: asynctask will always be created (and run) in a new thread,
          but if you want to spawn another one, set `True` here.
        :param loop: you may choose to create (and run) an asynctask in an existing loop.
        :return: an asynctask, which provides access to `id`, `coro`, `loop`.
        """
        cls.__ensure_init__()
        with cls.__mutex__:
            if new_thread and loop is not None:
                WARN(f'Task run in a new thread will have a new loop. (arg loop ignored: {loop})')
            if asyncio.iscoroutine(task_or_coro):
                if loop is None:
                    loop = cls.__instance__.current_loop if not new_thread else cls.append_new_loop()
                task = cls.create_task(task_or_coro, new_thread=new_thread, loop=loop)
            elif isinstance(task_or_coro, asyncio.Task):
                task = task_or_coro
                task_loop = getattr(task, 'loop', None)
                task_coro = getattr(task, 'coro', None)  # Py3.8 implemented task.get_coro()
                if task_coro is not None and (new_thread or (loop is not None and task_loop is not None and loop != task_loop)):
                    task = cls.create_task(task_coro, new_thread=new_thread, loop=loop)
                    WARN('Task is requested to run in a new thread or loop. a new task will be created.')
                    loop = task.loop
                else:
                    loop = task_loop or cls.__instance__.current_loop
                # if new_thread:
                #     WARN('Task is always bound with a loop and cannot be run in new thread. (arg ignored)')
                # if loop is not None and loop != task_or_coro.loop:
                #     WARN(f'Task is always bound with an existing loop. (arg loop ignored: {loop})')
            else:
                raise TypeError(f'Only accept coro object or task, while get a {type(task_or_coro)}')
            # loop.create_task() after loop.run_forever() will not be run, unless activate a `call_soon` for a new batch.
            if getattr(loop, 'id', None) == AsyncLoop.Main:
                # loop.call_soon(task)  # will be pending unless main_loop is running
                loop.run_until_complete(task)
            else:
                loop.call_soon_threadsafe(lambda: {})
            return task

    @classmethod
    def cancel_task(cls, id):
        """
        May directly call get_task(id).cancel() instead.
        """
        cls.__ensure_init__()
        with cls.__mutex__:
            task = cls.get_task(id)
            if task is None:
                raise ValueError(f'Task to be cancelled does not exist: {id}')
            return task.cancel()

    @classmethod
    def get_task(cls, id) -> asyncio.Task:
        cls.__ensure_init__()
        with cls.__mutex__:
            # for task in asyncio.Task.all_tasks(loop=cls.__it__.loop): ...
            try:
                task = cls.__instance__.all_tasks[id]
            except KeyError:
                task = None
            return task

    @classmethod
    def wait_task(cls, id, timeout):
        """
        :return: return `task.result()` if done before timeout, or `None` otherwise.
        """
        cls.__ensure_init__()
        with cls.__mutex__:
            task = cls.get_task(id)
            if task is None:
                raise ValueError(f'Task to be cancelled does not exist: {id}')
            loop = asyncio.get_event_loop()
            async def coro_simple_wait(task, timeout):
                return await asyncio.wait({task}, timeout=timeout)
            done, pending = loop.run_until_complete(coro_simple_wait(task, timeout))
            if task in done:
                return task.result()
            else:
                return None  # task is still pending

    @classmethod
    def mark_task(cls, id, purge=True):
        cls.__ensure_init__()
        with cls.__mutex__:
            task = cls.get_task(id)
            if task is not None:
                setattr(task, 'purge', purge)

    @classmethod
    def purge_tasks(cls):
        cls.__ensure_init__()
        with cls.__mutex__:
            for task_id, task in enumerate(cls.__instance__.all_tasks):
                if getattr(task, 'purge', False):
                    del cls.__instance__.all_tasks[task_id]


def amend_blank_cbs(cbs):
    """
    A helper method which can be used as follows, needn't to check if any part is `None` or not:
    ```
    on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
    try:
        while ...:
            ...
            on_progress()
        on_done()
    except:
        on_failed()
    finally:
        on_done()
    ```
    :param cbs: tuple of pre-defined callbacks.
    :return: a tuple in order of (on_done, on_succeeded, on_failed, on_progress) tuple, each part
      will be a blank callable unless it has been pre-defined by `cbs`.
    """
    amend_cbs = [lambda *a, **k: {}] * 4
    if hasattr(cbs, '__len__'):
        for i, cb in enumerate(cbs):
            amend_cbs[i] = cb
    return amend_cbs
