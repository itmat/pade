import redisconfig

# The URL of the broker. By default we use Redis on localhost, using
# the Redis's default port of 6380. If you change this, you must make
# sure you change the redis configuration.
BROKER_URL = 'redis://localhost/' + str(redisconfig.DB_CELERY)

CELERY_RESULT_BACKEND = 'redis'

