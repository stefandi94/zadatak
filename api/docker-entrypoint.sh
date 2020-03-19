gunicorn api.wsgi:application --timeout 600 --bind 0.0.0.0:8000
exec "$@"
