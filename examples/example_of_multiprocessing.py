#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import multiprocessing as mp


def job(q, v1, v2):
    x = 0
    x = v1 + v2
    print('job')
    q.put(x)

def job_1(q, v1, v2):
    x = 0
    x = v1 + v2
    print('job')
    return x

def job_2(v1, v2, l):
    l.acquire()
    x = v1-v2
    print(f'job,{x}')
    l.release()
def multicor_1():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q, 1, 2))
    p2 = mp.Process(target=job, args=(q, 3, 4))
    p1.start()
    p2.start()
    p2.join()
    p1.join()

    res1 = q.get()
    res2 = q.get()
    print(res1 + res2)

def multicore_2():
    pool = mp.Pool(processes=3)
    res = pool.map(job, (range(10), range(10)))
    print(res)
    res = pool.apply_async()
    res.get()

def multicore_3():
    val = mp.Value('d', 1)
    arr = mp.Array('i', [range(10)])

def multicore_4():
    lock = mp.Lock()
    val1 = mp.Value('i', 1)
    val2 = mp.Value('i', 2)
    p1 = mp.Process(target=job_2, args=(val1, val2, lock))
    p2 = mp.Process(target=job_2, args=(val1, val2, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == '__main__':
    multicore_4()