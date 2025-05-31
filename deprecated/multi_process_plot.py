import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Array, Lock, Manager, Pool, Queue
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import imageio
import time
import numpy as np
from tqdm import tqdm

class painter():
    def __init__(self, num_concurrent, steps=100):
        self.num_concurrent = num_concurrent
        self.steps = steps
        self.tqdm = None
        self.chunk_size = int(self.steps/self.num_concurrent)

    def _draw_figure_one_stage(self, label):
        task = self._task_generate(label)
        total_num = len(task)
        result_list = []
        for i in range(total_num):
            x, y, _ = task[i]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y)
            ax.set_xlim(0, self.steps)
            ax.set_ylim(0, self.steps)
            fig.savefig(f"./tmp/thread_{label}.png")
            plt.close(fig)
            result_list.append(imageio.imread(f"./tmp/thread_{label}.png"))
        print(f"!!!!!!!!! {label} !!!!!!!!!!!")
        # queue.put(result_list)
        return result_list

    def _task_generate(self, label):
        x = list(range(0,self.steps))
        y = list(range(0,self.steps))
        tmp = int((self.steps+label)/self.num_concurrent)
        print(tmp)
        if tmp == self.chunk_size:
            start = label*self.chunk_size
            end = (label+1)*self.chunk_size
        else:
            start = self.steps-(self.num_concurrent-label)*tmp
            end = start+tmp
        return [(x[i],y[i],i) for i in range(start, end)]     
        
    def _draw_figure(self):
        # queue = Queue()
        # tasks = [Process(target=self._draw_figure_one_stage, args=(queue,i)) for i in range(self.num_concurrent)]
        p = Pool(self.num_concurrent)
        ret = p.map(self._draw_figure_one_stage,range(self.num_concurrent))
        results = []
        for _, result in enumerate(ret):
            results += result
        print(len(results))
        imageio.mimsave("test.gif", results, 'GIF', duration=0.01)  

if __name__ == "__main__":
    start_time = time.time()
    P = painter(16,1000)
    P._draw_figure()
    end_time = time.time()
    print(f"total_time:{end_time-start_time}")
    math.sqrt(GM(2/r-1/a))*r = r1*norm(v)*math.sin(theta)
    
    