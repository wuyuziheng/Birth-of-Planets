import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import imageio
import time
import numpy as np
from tqdm import tqdm

class painter():
    def __init__(self, steps=100):
        self.steps = steps
        self.lock = None
        self.result_lock = None
        self.result_list = None
        self.tqdm = None
        self.init_time_list = None
        self.save_time_list = None

    def _draw_figure_one_stage(self, x, y, i, label):
        
        # print(f"start:{label}")
        start_time = time.time()
        # print(start_time)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        ax.set_xlim(0, self.steps)
        ax.set_ylim(0, self.steps)
        with self.lock[label]:
            check_time = time.time()
            fig.savefig(f"./tmp/thread_{label}.png")
            plt.clf()
            plt.close(fig)
            tmp_result = imageio.imread(f"./tmp/thread_{label}.png")
            end_time = time.time()
            with self.result_lock:
                # print(f"complete:{label}")
                self.tqdm.update(1)
                # self.result_list[i] = imageio.imread(f"./tmp/thread_{label}.png")
                self.result_list[i] = tmp_result
                self.init_time_list[label] += check_time - start_time
                self.save_time_list[label] += end_time - check_time

    def _draw_figure(self, num_concurrent):
        x = list(range(0,self.steps))
        y = list(range(0,self.steps))
        self.init_time_list = [0]*num_concurrent
        self.save_time_list = [0]*num_concurrent
        self.lock = [threading.Lock() for _ in range(num_concurrent)]
        self.result_lock = threading.Lock()
        self.tqdm = tqdm(total=self.steps)
        self.result_list = [None]*self.steps
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            tasks = [executor.submit(self._draw_figure_one_stage, x[i], y[i], i, i%num_concurrent) for i in range(self.steps)]
        # print(len(self.result_list))
        imageio.mimsave("test.gif", self.result_list, 'GIF', duration=0.01)  
        self.tqdm.close()

if __name__ == "__main__":
    num_concurrent = 16
    start_time = time.time()
    P = painter()
    P._draw_figure(num_concurrent)
    end_time = time.time()
    print(f"total_time:{end_time-start_time}")
    for i in range(num_concurrent):
        print(f"channel-{i}: init_time: {P.init_time_list[i]} save_time: {P.save_time_list[i]}")
    