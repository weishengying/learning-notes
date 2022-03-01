# mutex
[Mutex Doc](http://www.cplusplus.com/reference/mutex/mutex/)
mutex锁解决竞争问题，即某部分代码在同一时刻只允许一个线程访问。

```c++
// mutex example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex

std::mutex mtx;           // mutex for critical section

void print_block (unsigned long long n, char c) {
  // critical section (exclusive access to std::cout signaled by locking mtx):
  mtx.lock();
  for (int i=0; i<n; ++i) { }
  std::cout << "done\n";
  mtx.unlock();
}

int main ()
{
  std::thread th1 (print_block,5000000000000000000,'*');
  std::thread th2 (print_block,5000000000000000000,'$');

  th1.join();
  th2.join();

  return 0;
}
```

上述代码中，构建两个线程th1，th2，两个线程都调用print_block函数，函数中加锁后，
保证了mtx.lock()和mtx.unlock()之间的代码在同一时刻，只允许被一个线程访问。称mtx.lock()和mtx.unlock()之间的代码
为保护区。
简单来说，只有得到锁的线程，才拥有访问保护区代码的权力。**而未得到锁的线程，处于休眠状态，并不会轮询**。

注意：未得到锁的线程并非是处于不断的尝试去得到锁的状态！而是处于休眠状态！

一个线程或者进程由于等待锁而被阻塞或挂起时，实际的操作通常是把当前进程或者线程从就绪队列移除，
挂在所等待的锁的等待队列上，同时切换到下一个被调度器选中的进程或者线程上；
释放时，等待队列上的所有进程和线程都会被移 回就绪队列，等待调度器的下次一调度。因此：
1. 是否就完全不占用CPU,还是还占用一点CPU,用来检测有没有获得锁?
  基本上不占用CPU，锁被释放时，队列上的进程或者线程会被唤醒，也就是说从锁的等待队列上移除，并加入到就绪队列，等待调度器的下一次调度。

2. 如果如1所说,那一个线程/进程被挂起,好像对系统的性能也没有多少影响,因为CPU被调度到其他进程了,也没有浪费(不考虑锁的消耗),主要浪费是指上下文的切换吗?
  确实没什么影响，只不过进程的切换都会有上下文（也就是寄存器和页表等）的切换。

3. Linux的互斥锁一般是硬件实现吗,还是比如说while(mute == 1){...}不停的取mutex的值?
  你说的的互斥锁应该是自旋锁这种吧。自旋锁等待时一般都是通过硬件支持或者软件模拟的原子操作以类似你说的这种方式去操作锁的一个标志，并在两次等待之间保持中断使能，从而使得其他进程和线程能够得到执行机会。
[参考文档](https://mbd.baidu.com/ug_share/mbox/4a81af9963/share?tk=08d1d7c3926067dddcc7f579dcfa7355&share_url=https%3A%2F%2Fwjrsbu.smartapps.cn%2Fzhihu%2Fanswer%3Fid%3D2009359020%26isShared%3D1%26_swebfr%3D1%26_swebFromHost%3Dbaiduboxapp)

## 验证
编译上述代码（g++ demo.cc -lpthread），验证未得到锁的线程，是否处于休眠状态。
利用top -H  命令(top -H -d 1| grep Threads)，查看当前系统进程状态。

> Threads: 173 total,   1 running, 172 sleeping,   0 stopped,   0 zombie

运行a.out

> Threads: 176 total,   2 running, 174 sleeping,   0 stopped,   0 zombie
> Threads: 175 total,   1 running, 173 sleeping,   0 stopped,   0 zombie

运行之后，main主线程加上子线程th1，th2，总线程+3，runing+1，sleep+2（主线程和未得到锁的自线程休眠）；
主线程休眠的原因是因为th1.join();th2.join()(注：该demo中主线程必须等待子线程结束后，才能结束，否则主线程先结束，各子线程也会立马结束);主线程需要等待子线程执行完毕，等待期间，主线程处于休眠状态。
一个自线程运行完后退出，总线程-1，sleep-1（只剩主线程休眠）

> Threads: 173 total,   1 running, 172 sleeping,   0 stopped,   0 zombie

全部于行完之后，恢复原来的状态。


# unique_lock
[Unique_lock Doc](http://www.cplusplus.com/reference/mutex/unique_lock/)
unique_lock是一个锁的管理类：A unique lock is an object that manages a mutex object with unique ownership in both states: locked and unlocked.
Unique_lock的构造函数中可以调用lk.lock()，详细信参考[std::unique_lock::unique_lock](http://www.cplusplus.com/reference/mutex/unique_lock/unique_lock/);析够时调用lk.unlock();
最简单常见的应用是利用局部变量，隐式调用lock, unlock,减少代码量。
```cpp
// unique_lock example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex

std::mutex mtx;           // mutex for critical section

void print_block (int n, char c) {
  // critical section (exclusive access to std::cout signaled by lifetime of lck):
  std::unique_lock<std::mutex> lck (mtx);
  for (int i=0; i<n; ++i) { }
  std::cout << "done\n";
}

int main ()
{
  std::thread th1 (print_block,5000000000000000000,'*');
  std::thread th2 (print_block,5000000000000000000,'$');

  th1.join();
  th2.join();

  return 0;
}
```
效果和第一个mutex demo是一样的。

# lock_guard
[Lock_guard Doc](http://www.cplusplus.com/reference/mutex/lock_guard/)
lock_guard也是一个锁的管理类，使用上与unique_lock类似。