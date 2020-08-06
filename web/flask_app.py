from helpers.util import DEBUG, INFO, WARN, ERROR, get_new_name_if_exists, Params, ensure_dir_exists
from typing import Callable, Iterable

__all__ = [
    "get_webapp",
]

# IMPROVE: use map cache: (package_name, root_path) -> webapp instance
app = None

from shutil import copyfile
from flask import Flask, request, url_for


class WebApp(Flask):
    EventUploads = 'uploads'
    EventTaskQuery = 'task_query'

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.handlers = set()  # element: (event_name, namespace, handler(=callable), is_onetime)

    # --- for registry/subscription ---------------
    def on(self, event_name: str, namespace=None, onetime=False):
        """Decorator to define a custom handler for web app events.
        The handler function must accept one argument. Example::

            @app.on('uploads', namespace='/type/name')
            def uploads_handler(e):
                paths = e.data.paths

        :param event_name: The name of the event. Reservered event: ``'uploads'``,
                        ``'connect'`` and ``'disconnect'``
        :param namespace: The namespace for which to register the handler.
                        Defaults to the global namespace.
        :param onetime: registered handler will only be called once then obsolete
        """
        namespace = namespace or '/'

        def decorator(handler):
            if not callable(handler):
                raise ValueError('handler must be callable')
            self.handlers.add((event_name, namespace, handler, onetime))
            return handler

        return decorator

    def on_uploads(self, namespace=None, onetime=False) -> Callable:
        """
        Arguments of a handler: filepath_or_list: (str, list)
        Return value of a handler: a dict. keys include `asynctask_id`, `filename` alike.
        """
        return self.on(event_name=self.__class__.EventUploads, namespace=namespace, onetime=onetime)

    def on_task_query(self, namespace=None, onetime=False) -> Callable:
        """
        Arguments of a handler: task_id
        Return value of a handler: task status and result
        """
        return self.on(event_name=self.__class__.EventTaskQuery, namespace=namespace, onetime=onetime)

    # --- for dispatch to subscribers ---------------
    def dispatch_handlers(self, event_name: str, *args, namespace=None, **kwargs):
        """
        :param event_name:
        :param args:
        :param namespace: if None means needn't to compare namespace
        :param kwargs:
        :return:
        """
        handler_results = []
        to_delete = set()
        # IMPROVE: use mutex to prevent handlers add/delete during processing
        handlers = self.handlers.copy()
        for _event_name, _namespace, _handler, _is_onetime in handlers:
            if _event_name == event_name and (namespace is None or _namespace == namespace):
                try:
                    # import inspect
                    # DEBUG(f"_handler signature: ({[param.kind.description for param in inspect.signature(_handler).parameters.values()]})")
                    DEBUG(f"[{_event_name}{'@'+(_namespace or '')}] dispatch({args}, {kwargs})")
                    if _is_onetime:
                        to_delete.add((_event_name, _namespace, _handler, _is_onetime))
                    handler_result = _handler(*args, **kwargs)
                    handler_results.append(handler_result)
                except Exception as e:
                    WARN(f"Registered handler caused exception ({_event_name}@{_namespace}, "
                         f"which should have been caught in handler side): {e}")
        self.handlers -= to_delete
        return handler_results

    # --- other functions ---------------
    @staticmethod
    def hack_webapp(webapp, host, port):
        if host is None or port is None:
            host, _, port = webapp.config['SERVER_NAME'].partition[':']
        setattr(webapp, 'host', host)
        setattr(webapp, 'port', port)

    # override
    def run(self, *args, **kwargs):
        self.__class__.hack_webapp(self, kwargs.get('host', None), kwargs.get('port', None))
        # blocking method
        super(self.__class__, self).run(*args, **kwargs)

    def async_run(self, **params):
        """
        Generally we need to launch web app in another loop/thread, to not block ML operations.
        """
        webapp = self
        self.__class__.hack_webapp(webapp, params.get('host', None), params.get('port', None))

        # IMPROVE: web app need not to run in an aysncio loop (host in a new thread), to run in a new thread is enough.
        from async_ import AsyncLoop, AsyncManager

        async def coro_webapp_run(): webapp.run(**params)
        webapp_loop = AsyncManager.get_loop(AsyncLoop.WebApp)
        task = AsyncManager.run_task(coro_webapp_run(), loop=webapp_loop)
        DEBUG(f"[webapp_loop] listening to port {params.get('port', '<unknown>')} ...")
        return task

    @staticmethod
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()


def get_webapp(import_name=__name__, root_path=None, **params):
    """
    Note that if an webapp instance already exists, if will be reused with arguments ignored.
    """
    from flask import flash, request, redirect, url_for, render_template, send_from_directory, send_file, make_response
    import urllib.request
    from werkzeug.utils import secure_filename
    import json
    import os.path as osp
    global app

    # FIXME: import_name attr in app.config? => if yes, app.config.get('import_name', None) == import_name
    if isinstance(app, Flask) and (root_path is None or app.root_path == root_path):
        return app
    elif app is not None:
        WebApp.shutdown()  # close existing web app

    app = WebApp(import_name, root_path=root_path)
    # -- non-configurable ----
    app.secret_key = "mice love rice"
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    RESPONSE_JSON_ACCESSCONTROL = {'Content-type': 'application/json; charset=utf-8',
                                   'Access-Control-Allow-Origin': '*',
                                   'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                                   'Access-Control-Allow-Headers': 'x-requested-with, X-PINGOTHER, Content-Type'}
    # -- configurable --------
    # NOTE: a relative path will be handled as rooted from web app's root path
    app.config['UPLOAD_FOLDER'] = params.get('upload_folder', 'static/uploads')

    @app.errorhandler(404)
    def not_found(error):
        # return f"Yet another page not found: {error}"
        return json.dumps({'error': 'Not found'}), 404, {'Content-Type': 'application/json; charset=utf-8'}

    @app.route('/samples/upload_form_ui')
    def samples_upload_form_ui():
        return render_template('samples/upload_form_ui.html')

    @app.route('/samples/upload_dnd_ui')
    def samples_upload_dnd_ui():
        # return render_template('samples/upload_dnd_ui.html')
        return redirect(url_for('static', filename='samples/upload_dnd_ui.html'), code=301)

    @app.route('/api/0.1/uploads/<string:filename>', methods=['GET'])
    def upload_retrieve(filename):
        attach = request.values.get('attach', default=None) is not None
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=attach)

    @app.route('/samples/test', methods=['POST'])
    def samples_test():
        return redirect(url_for('test'), code=301)

    @app.route('/api/0.1/uploads', methods=['POST'])
    @app.route('/api/0.1/uploads/<string:filename_to_update>', methods=['POST'])
    def upload_create_or_update(filename_to_update=None, key="file"):
        """
        :return: json object. contains relative filename on success, and error message on failure.
        """
        # redirect_url = request.values.get('redirect', default=request.url, type=str)

        # 1. request -> files(data) -> local uploads folder + json response(error+filename)
        # Accept multiple files
        # file = request.files[key]
        files = request.files.getlist(key)
        if files is None or len(files) == 0:
            ret = {'error': 'no file part found in multipart/form-data'}
            return str(json.dumps(ret)), 400, RESPONSE_JSON_ACCESSCONTROL
        # NOTE: use [] * len(..) carefully.. it just do soft copy. use `for` instead.
        ret = [{} for _ in range(len(files))]  # [{filename: str, error: optional(str)}]
        dispatch_arg = []

        error_count = 0
        for idx, file in enumerate(files):
            if file.filename == "":
                ret[idx].update({'error': "no file name is given or no file selected for uploading"})
                error_count += 1
                continue  # bypass to the next one

            if file and osp.splitext(file.filename)[1].lower() in ALLOWED_EXTENSIONS:
                if filename_to_update is None:
                    # TODO: handle chinese filename. str.encode('utf-8')?
                    filepath = secure_filename(file.filename)
                    filepath = get_new_name_if_exists(osp.join(app.config['UPLOAD_FOLDER'], filepath))
                else:
                    filepath = osp.join(app.config['UPLOAD_FOLDER'], filename_to_update)
                if not osp.isabs(filepath):
                    filepath = osp.join(app.root_path, filepath)
                try:
                    file.save(filepath)  # NOTE: overwrite existed one
                except Exception as e:
                    ret[idx].update({'error': f"Failed to upload file to {filepath}"})
                    error_count += 1
                    continue  # bypass to the next one
                INFO('file uploaded to: ' + filepath)
                dispatch_arg.append(filepath)
                ret[idx].update({'filename': osp.basename(filepath)})
            else:
                ret[idx].update({'error': f"only accept these image types: {ALLOWED_EXTENSIONS}"})
                error_count += 1
                continue  # bypass to the next one
        ret = {'uploaded': ret}

        # 2. dispatch to subscribers of `on_uploads` event
        if error_count < len(files):  # error_count == 0:
            dispatch_results = app.dispatch_handlers(app.__class__.EventUploads,
                                                    dispatch_arg if len(dispatch_arg) > 1 else dispatch_arg[0])
            # NOTE: multiple inputs can be consumed by once, so results num can be less than inputs num.
            ret.update({'dispatched:': dispatch_results})

        return str(json.dumps(ret)), 200 if error_count < len(files) else 400, RESPONSE_JSON_ACCESSCONTROL

    @app.route('/api/0.1/tasks/<string:id>', methods=['GET'])
    def task_query(id):
        ret = {}  # progress, result, error
        from async_ import AsyncManager
        task = AsyncManager.get_task(id)
        if task is not None:
            ret.update({'progress': getattr(task, 'progress', 0)})
            if task.done():
                if task.cancelled():
                    ret.update({'error': 'cancelled'})
                elif task.exception() is not None:
                    ret.update({'error': task.exception().args[0]})
                else:
                    ret.update({'result': task.result()})
        else:
            ret.update({'error': 'not found'})
        return ret

    @app.route('/api/0.1/tasks/<string:id>', methods=['DELETE'])
    def task_delete(id):
        from async_ import AsyncManager
        task = AsyncManager.get_task(id)
        if task is not None:
            task.cancel()
        # IMPROVE: wait a while and return result of cancelling
        # return ret

    # TEMP: merge to task_query() later
    @app.route('/api/0.1/current_task', methods=['GET'])
    def task_query_current():
        ret = {}  # status, result, error
        # return finished only when the subscriber finished its current task
        dispatch_arg = None  # TODO: task_id
        dispatch_results = app.dispatch_handlers(app.__class__.EventTaskQuery, dispatch_arg)
        if len(dispatch_results) == 0:
            ret.update({'status': 'not_found'})
        else:
            dispatch_result = dispatch_results[0]  # TEMP
            assert isinstance(dispatch_result, dict), f"dispatch_results is expected to be a dict, but get {type(dispatch_result)}"
            # expected values: 1.{'status': 'processing'} 2.{'status': 'finished', 'result': '/experiments/.../...jpg'}
            if dispatch_result.get('status', None) == 'finished':
                # convert dispatch_result['result'] from abspath to url, including file copy
                src_abspath = dispatch_result['result']
                filename = osp.basename(src_abspath)
                # copy abspath -> subfolder under 'static/'
                dest_folder = 'history'
                dest_abspath = osp.join(app.static_folder, dest_folder)
                ensure_dir_exists(dest_abspath)
                dest_abspath = osp.join(dest_abspath, filename)
                copyfile(src_abspath, dest_abspath)
                url = f'{app.static_url_path}/{dest_folder}/{filename}'
                dispatch_result.update({'result': url})
            ret.update(dispatch_result)
        return ret

    @app.route('/hello/<string:message>')
    def say_hello(message=None):
        return f"hello world -- from Flask! {message}"

    @app.route('/')
    def root():
        # return "A Pure Flask-based Web Site!"
        return redirect(url_for('samples_upload_dnd_ui'), code=301)

    return app
