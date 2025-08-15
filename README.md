军训大白兔蹦出来了潍坊吃雪白的扔子视频大全免费


enum cudaMemcpyKind kind,
cudaStream_t stream
);

也就是说，cudaMemcpyAsync 只比 cudaMemcpy 多一个参数。该函数的最后一个参数就是所在流的变量。

        在使用异步的数据传输函数时，需要将主机内存定义为不可分页内存（non-pageable memory）或者固定内存（pinned memory）。不可分页内存是相对于可分页内存（pageable memory）的。操作系统有权在一个程序运行期间改变程序中使用的可分页主机内存的物理地址。相反，若主机中的内存声明为不可分页内存，则在程序运行期间，其物理地址将保持不变。如果将可分页内存传给 cudaMemcpyAsync 函数，则会导致同步传输，达不到重叠核函数执行与数据传输的效果。主机内存为可分页内存时，数据传输过程在使用 GPU 中的 DMA 之前必须先将数据从可分页内存移动到不可分页内存，从而必须与主机同步。主机无法在发出数据传输的命令后立刻获得程序的控制权，从而无法实现不同 CUDA 流之间的并发。

相关介绍可以看以前的博客

虚拟内存、内存分段、分页、CUDA编程中的零拷贝_cuda零拷贝内存-CSDN博客

不可分页主机内存的分配可以由以下两个 CUDA 运行时 API 函数中的任何一个实现：

cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, size_t flags);

注意，第二个函数的名字中没有字母 M。若函数 cudaHostAlloc 的第三个参数取默认
值 cudaHostAllocDefault，则以上两个函数完全等价。由以上函数分配的主机内存必须由如下函数释放：cudaError_t cudaFreeHost(void* ptr);如果不小心用了 free 函数释放不可分页主机内存，会出现运行错误。
4.2  重叠核函数执行与数据传输的例子
