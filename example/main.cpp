
#include <vector>
#include <iostream>
#include <memory>
#include <EC.hpp>

using namespace std;
using namespace EC;





struct T{};

int main(){
    AT::register_all_kernels();
    AT::Tensor a = AT::Tensor::normal({2,2});
    AT::Tensor b = AT::Tensor::normal({2,1});
    auto c = AT::gemv(a,b,AT::Tensor::ones({2,1}),2,1);

    a.print(6,6);
    b.print(6,6);
    c.print(6,6);

    // auto [g, out] = Tr::trace([&] {
    // auto x = Tr::input({2, 3}, DType::Float32, "x");
    // auto y = Tr::input({2, 3}, DType::Float32, "y");
    //     return AT::add(x, y);
    // });

}