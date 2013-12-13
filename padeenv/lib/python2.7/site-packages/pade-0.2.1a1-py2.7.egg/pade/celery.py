"""Sets up Celery for PADE."""

from __future__ import absolute_import, print_function, division

from celery import Celery

celery = Celery('pade.celery',
                include=['pade.tasks'])
celery.config_from_object('celeryconfig')

if __name__ == '__main__':
    celery.start()
