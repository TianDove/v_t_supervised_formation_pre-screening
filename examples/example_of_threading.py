#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import threading
import time
import queue
import sys


def thread_job():
    print(f'T1 Start.')
    for i in range(10):
        # 当前线程信息
        print(f'Added Thread, ID:{threading.currentThread()}')
        time.sleep(1)
    print('T1 Finished.')

def thread_job_T2():
    print(f'T2 Start.')
    for i in range(10):
        # 当前线程信息
        print(f'Added Thread, ID:{threading.currentThread()}')
        time.sleep(0.5)
    print('T2 Finished.')

def main():
    """"""
    # 当前活动中的线程数
    print(threading.active_count())
    # 列出当前活动中的线程
    print(threading.enumerate())
    # 添加新线程，target：指定任务，name：线程命名
    added_thread = threading.Thread(target=thread_job, name='T1')
    another_thread = threading.Thread(target=thread_job_T2(), name='T2')
    # 开始线程
    added_thread.start()
    another_thread.start()
    # added_thread 运行完后才运行join后的部分
    added_thread.join()
    another_thread.join()
    #
    print('All Thread Done.')

def job(l, q):
    for i in range(len(l)):
        l[i] = l[i]**2

    # 结果存入队列
    q.put(l)

# 主线程
def main_2():
    q = queue.Queue()
    thread_list = []
    data = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
    # 多个子线程
    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q), name=f'T_{i}')
        t.start()
        thread_list.append(t)

    # 主线程等待所有子线程结束
    for thread in thread_list:
        thread.join()

    results = []
    # 按队列顺序取结果
    for val in range(4):
        results.append(q.get())

    print(results)

def job1():
    global A, lock
    lock.acquire()
    for i in range(4):
        A += 1
        print('job1', A)
    lock.release

def job2():
    global A, lock
    lock.acquire()
    for i in range(4):
        A += 10
        print('job2', A)
    lock.release

def main_3():
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    main_3()
    sys.exit(0)