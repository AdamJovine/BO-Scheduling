from app import create_app
#from pinned.routes import server_bp
app = create_app()

if __name__ == '__main__':
    # for dev
    app.run(host='0.0.0.0', port=5000, debug=True)
