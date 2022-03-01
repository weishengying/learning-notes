# std::this_thread::yield
简而言之，调用该函数的线程会让出时间片，重新进行调度。

demo
```cpp
// this_thread::yield example
#include <iostream>       // std::cout
#include <thread>         // std::thread, std::this_thread::yield
#include <atomic>         // std::atomic

std::atomic<bool> ready (false);

void count1m(int id) {
  while (!ready) {             // wait until main() sets ready...
    std::this_thread::yield();
  }
  for (volatile unsigned long long i=0; i<1000000000; ++i) {}
  std::cout << id << "\n";
}

int main ()
{
  std::thread threads[10];
  std::cout << "race of 10 threads that count to 100 million:\n";
  for (int i=0; i<10; ++i) threads[i]=std::thread(count1m,i);
  ready = true;               // go!
  for (auto& th : threads) th.join();
  std::cout << '\n';

  return 0;
}
```
十个线程同时比赛数数，ready信号相当于开始信号，主线程设置好ready后，子进程才开始比赛，在主线程设置好ready之前，子进程通过调用yeild函数，让出时间片，不占用cpu。
运行前线程状况：

> Threads: 173 total,   1 running, 171 sleeping,   0 stopped,   0 zombie

运行时：

> Threads: 183 total,  11 running, 172 sleeping,   0 stopped,   0 zombie

十个子线程开始竞赛，主线休眠等待，所以running+10， sleeping+1

进一步验证，ready之前，yield()函数让子线程让出时间片，真的不占用cpu吗
验证代码：
```cpp
// this_thread::yield example
#include <iostream>       // std::cout
#include <thread>         // std::thread, std::this_thread::yield
#include <atomic>         // std::atomic

std::atomic<bool> ready (false);

void count1m(int id) {
  while (!ready) {             // wait until main() sets ready...
    std::this_thread::yield();
  }
  for (volatile unsigned long long i=0; i<1000000000; ++i) {}
  std::cout << id << "\n";
}

int main ()
{
  std::thread threads[10];
  std::cout << "race of 10 threads that count to 100 million:\n";
  for (int i=0; i<10; ++i) threads[i]=std::thread(count1m,i);
  for (volatile unsigned long long i=0; i<10000000000; ++i) {}
  ready = true;               // go!
  for (auto& th : threads) th.join();
  std::cout << '\n';

  return 0;
}
```
在主线程设置ready之前，主线程做一些工作（比如也计数），然后观察子线程是否占用了cpu资源。
运行前线程情况如下：
> Threads: 174 total,   1 running, 173 sleeping,   0 stopped,   0 zombie

运行demo后：

> Threads: 185 total,  12 running, 173 sleeping,   0 stopped,   0 zombie

runing+11
可见，子线程并未真正的让出时间片。原因可以参考：[std::this_thread::yield](https://en.cppreference.com/w/cpp/thread/yield)

**Notes**
The exact behavior of this function depends on the implementation, in particular on the mechanics of the OS scheduler in use and the state of the system. For example, a first-in-first-out realtime scheduler (SCHED_FIFO in Linux) would suspend the current thread and put it on the back of the queue of the same-priority threads that are ready to run (and if there are no other threads at the same priority, yield has no effect).

这样虽然不会影响性能，但会增加功耗。