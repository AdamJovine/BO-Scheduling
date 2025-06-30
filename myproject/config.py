import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SQLALCHEMY_DATABASE_URI  = os.environ.get('DATABASE_URL') \
                              or 'sqlite:///' + os.path.join(basedir, 'data.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SAVE_PATH = os.environ.get('SAVE_PATH') or os.path.join(basedir, 'data')
    UI_PATH   = '/home/asj53/BOScheduling/UI/pages/plots'#os.environ.get('UI_PATH')   or os.path.join(basedir, 'ui')
    NUM_SLOTS = int(os.environ.get('NUM_SLOTS', 24))
    CORS_ORIGINS = ["*"]
    print('SAVE_PATH' , SAVE_PATH)
    print('UI_PATH' , UI_PATH )
