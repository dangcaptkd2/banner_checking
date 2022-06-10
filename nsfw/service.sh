#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WORK_PATH=../nsfw
SIGNAL="image_nsfw.py"

echo "[.] Find service $JOB_NAME ...."

##
cd $WORK_PATH
printf "[.] Goto working path: $WORK_PATH \n"
export PYTHONPATH=$WORK_PATH


FIND_SERVICE=$(ps -ef | grep -v grep | grep -c "$SIGNAL")

case "$1" in
    stop)		
		if [ $FIND_SERVICE -gt 0 ]; then
			PROCESS_ID=$(ps -ef | grep -v grep | grep "$SIGNAL" | cut -c10-15)
            printf "[.] Service runing pid: ${GREEN} $PROCESS_ID ${NC}\n"
			kill $PROCESS_ID
			echo "[.] Stopped."
		else
			echo "[.] Service not found."
		fi
        ;;
	restart)				
		if [ $FIND_SERVICE -gt 0 ]; then
			PROCESS_ID=$(ps -ef | grep -v grep | grep "$SIGNAL" | cut -c10-15)
            printf "[.] Service runing pid: ${GREEN} $PROCESS_ID ${NC}\n"
			kill $PROCESS_ID
			echo "[.] Stopped."
		fi
		python3.7 $SIGNAL > /dev/null 2>&1 &
		echo "[.] Done."
        ;;
    *)  
        if [ $FIND_SERVICE -gt 0 ]; then
            PROCESS_ID=$(ps -ef | grep -v grep | grep "$SIGNAL" | cut -c10-15)
            printf "[.] Service runing pid: ${GREEN} $PROCESS_ID ${NC}\n"
        else            
            python3.7 $SIGNAL > /dev/null 2>&1 &
            echo "[.] Done"
        fi
		;;
esac
