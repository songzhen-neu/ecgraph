import _thread
import time
import threading

# 为线程定义一个函数
count={}
count_int=0

threadCount=0
threadNum=5


# _thread 0.30   thread 0.0078
def print_time():
    global count_int
    global threadCount

    for i in range(1000):
        count_int+=1
        print(count_int)
    # while True:
    #     pass
    threadCount+=1

# 创建两个线程
try:
    count[0]=0
    start=time.time()
    thread_list =[]
    for i in range(threadNum):
        thread_list.append(threading.Thread(target=print_time, args=( )))
        thread_list[i].start()
        thread_list[i].join()
        # _thread.start_new_thread( print_time, ( ) )
    #
    # _thread.start_new_thread( print_time, ( count,) )
    # _thread.start_new_thread( print_time, ( count,) )
    # _thread.start_new_thread( print_time, ( count,) )
    # _thread.start_new_thread( print_time, ( count,) )
    # _thread.start_new_thread( print_time, ( count,) )
    # t1=threading.Thread(target=print_time, args=(count, ))
    # t2=threading.Thread(target=print_time, args=(count, ))
    # t3=threading.Thread(target=print_time, args=(count, ))

    # for i in range(threadNum):
    #     thread_list[i].start()
    #     thread_list[i].join()

    while threadCount<threadNum:
        pass

    end=time.time()
    print("time:{0}".format(end-start))

except:
    print ("Error: 无法启动线程")

while 1:
    pass