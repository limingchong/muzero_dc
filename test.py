import time
import threading
import queue

class Worker(threading.Thread):
    def __init__(self, name, queue, delay):
        threading.Thread.__init__(self)
        threading.Thread()
        self.queue = queue
        self.delay = delay
        self.start()    #执行run()

    def run(self):
        #循环，保证接着跑下一个任务
        while True:
            # 队列为空则退出线程
            if self.queue == 0:
                break
            time.sleep(self.delay)
            # 打印
            print(self.getName() + " process " + str(self.queue))
            # 任务完成
            self.queue -= 1


# 队列
queue = 10
for i in range(2):
    threadName = 'Thread' + str(i)
    Worker(threadName, queue, 0.1)
