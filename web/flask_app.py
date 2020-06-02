from helpers.util import DEBUG, INFO, WARN, ERROR, get_new_name_if_exists

__all__ = [
    "get_webapp",
]

# IMPROVE: use map cache: (package_name, root_path) -> webapp instance
app = None

def get_webapp(import_name=__name__, root_path=None):
    from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file, make_response
    import urllib.request
    from werkzeug.utils import secure_filename
    import json
    import os.path as osp
    global app

    if isinstance(app, Flask) and app.import_name == import_name \
            and (root_path is None or app.root_path == root_path):
        return app

    app = Flask(import_name, root_path=root_path)
    app.secret_key = "mice love rice"
    app.config['UPLOAD_FOLDER'] = "static/uploads"
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    RESPONSE_JSON_ACCESSCONTROL = {'Content-type': 'application/json; charset=utf-8',
                                   'Access-Control-Allow-Origin': '*',
                                   'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                                   'Access-Control-Allow-Headers': 'x-requested-with, X-PINGOTHER, Content-Type'}

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

    # IMPROVE: start with '/api/<version>' for version control
    @app.route('/api/0.1/uploads/<string:filename>', methods=['GET'])
    def upload_get(filename):
        attach = request.values.get('attach', default=None) is not None
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=attach)

    @app.route('/api/0.1/uploads', methods=['POST'])
    @app.route('/api/0.1/uploads/<string:filename>', methods=['POST'])
    def upload_create(filename=None, key="file"):
        """
        :return: json object. contains relative filename on success, and error message on failure.
        """
        # redirect_url = request.values.get('redirect', default=request.url, type=str)

        # Accept multiple files
        # file = request.files[key]
        files = request.files.getlist(key)
        if files is None or len(files) == 0:
            ret = {'error': 'no file part found in multipart/form-data'}
            return str(json.dumps(ret)), 400, RESPONSE_JSON_ACCESSCONTROL
        ret = {'error': '', 'filename': ''} if len(files) == 1 else {'error': ['']*len(files), 'filename': ['']*len(files)}

        def set_map_entry(map, key, idx, data):
            if isinstance(map[key], list):
                map[key][idx] = data
            else:
                map[key] = data

        error_count = 0
        for idx, file in enumerate(files):
            if file.filename == "":
                set_map_entry(ret, 'error', idx, "no file name is given or no file selected for uploading")
                error_count += 1
                continue  # bypass to the next one

            if file and osp.splitext(file.filename)[1].lower() in ALLOWED_EXTENSIONS:
                if filename is None:
                    filepath = secure_filename(file.filename)
                    filepath = get_new_name_if_exists(osp.join(app.config['UPLOAD_FOLDER'], filepath))
                else:
                    filepath = osp.join(app.config['UPLOAD_FOLDER'], filename)
                if not osp.isabs(filepath):
                    filepath = osp.join(app.root_path, filepath)
                try:
                    file.save(filepath)  # NOTE: overwrite existed one
                except Exception as e:
                    set_map_entry(ret, 'error', idx, f"Failed to upload file to {filepath}")
                    error_count += 1
                    continue  # bypass to the next one
                INFO('file uploaded to: ' + filepath)
                set_map_entry(ret, 'filename', idx, osp.basename(filepath))
                continue
            else:
                set_map_entry(ret, 'error', idx, f"only accept these image types: {ALLOWED_EXTENSIONS}")
                error_count += 1
                continue  # bypass to the next one
        return str(json.dumps(ret)), 200 if error_count < len(files) else 400, RESPONSE_JSON_ACCESSCONTROL

    @app.route('/hello/<string:message>')
    def say_hello(message=None):
        return f"hello world -- from Flask! {message}"

    @app.route('/')
    def root():
        # return "A Pure Flask-based Web Site!"
        return redirect(url_for('samples_upload_dnd_ui'), code=301)

    return app

