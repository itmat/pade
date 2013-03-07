import redisconfig

BROKER_URL = 'redis://localhost/' + str(redisconfig.DB_CELERY)
CELERY_RESULT_BACKEND = 'redis'

