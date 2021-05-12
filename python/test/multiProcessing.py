from multiprocessing import Process, Queue
import numpy as np
import time


def f(q, i,t):
    # if (i == 3):
    #     time.sleep(2)
    a = []
    for i in range(10):
        a.append(list())
        for j in range(10):
            a[i].append(5)

    print(t)
    b = np.array(a)
    # print(type(b))
    q.put(b)
    # while True:
    #     pass

    # if __name__ == '__main__':
q = []
jobs = []
for t in range(100):
    for i in range(5):
        q.append(Queue())
        p = Process(target=f, args=(q[i], i,t,))
        jobs.append(p)
        p.start()

    for i in range(5):
        # print(q[i].get())
        q[i].get()
        jobs[i].terminate()


    # while not q.empty():
    #     print(q.get())    # prints "[42, None, 'hello']"
