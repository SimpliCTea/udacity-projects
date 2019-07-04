from app import app

if __name__ == '__main__':
    app.run()

# run app with
# export FLASK_APP=run.py !on windows use set instead of export
# export FLASK_ENV=development
# flask run --without-threads