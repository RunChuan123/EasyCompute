
#include <vector>
#include <iostream>
#include <memory>
#include <EC.hpp>

using namespace std;
using namespace EC;





struct T{};

int main(){
    EC_INIT();

    AT::Tensor a = AT::Tensor::normal({2,3});
    
    AT::Tensor b = AT::Tensor::normal({2,3});

    auto c = a + b;
    // AT::add_into
    a.print(6,6);
    b.print(6,6);
    c.print(6,6);
    // AT::CPU::add_into_imlp(a,b,c);
    // c.print(6,6);





    // auto [g, out] = Tr::trace([&] {
    // auto x = Tr::input({2, 3}, DType::Float32, "x");
    // auto y = Tr::input({2, 3}, DType::Float32, "y");
    //     return AT::add(x, y);
    // });

}