from infrastructure.di.container import Container as container
from infrastructure.di import logger
logger.log()





def main()->None:
    container().\
        get_usecase().\
        execute()

if __name__=='__main__':
    main()

