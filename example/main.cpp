
#include <vector>
#include <iostream>
#include <memory>
#include <EC.hpp>

using namespace std;
using namespace EC;





struct T{};

int main(){
    AT::register_all_kernels();
    AT::Tensor a = AT::Tensor::normal({3,4});
    AT::Tensor b = AT::Tensor::normal({3,4});
    // Tensor c = a * b;
    // Tensor d = sin(c);
    // a.print(6,6);
    // b.print(6,6);
    // c.print(6,6);
    // d.print(6,6);
// [ [ -0.856571 1.342388 1.184361 -1.230271 ],
//   [ 0.739764 -0.037925 1.433669 -1.924589 ],
//   [ 0.868918 -0.065477 -0.695220 2.141394 ] ]
// [ [ 0.155353 -1.849660 -0.010045 2.199045 ],
//   [ -0.853364 1.214944 -0.474211 0.125389 ],
//   [ 1.352113 0.831725 -0.256672 -0.885722 ] ]
// [ [ -0.133071 -2.482961 -0.011896 -2.705422 ],
//   [ -0.631288 -0.046076 -0.679862 -0.241323 ],
//   [ 1.174876 -0.054459 0.178444 -1.896680 ] ]
// [ [ -0.132678 -0.612035 -0.011896 -0.422472 ],
//   [ -0.590185 -0.046060 -0.628686 -0.238988 ],
//   [ 0.922642 -0.054432 0.177498 -0.947368 ] ]
     auto tc = Tr::trace([&]{
        AT::Tensor c = AT::add(a, b); // 这里会录图，不会执行
        // tc.g.outputs 你可以在 trace 结束后设置，也可以在 API 里 return tensor 来自动设置
    });

    // 假设你把 outputs 设置成最后那个张量的 ValueId
    // 这里简化：你需要在 trace API 里捕获返回值作为 graph.outputs

    Gr::Executor ex;
    std::unordered_map<Gr::ValueId, AT::Tensor> feeds;
    // feeds[graph_input_id] = ...
    auto out = ex.run(tc.g, feeds);
}