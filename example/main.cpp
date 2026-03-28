
#include <vector>
#include <iostream>
#include <memory>
#include <EC.hpp>

using namespace std;
using namespace EC;





struct T{};

int main(){
    EC_INIT();
// ,3.0,2.0,DType::f16
    AT::Tensor a = AT::Tensor::normal({2,3});
    
    AT::Tensor b = AT::Tensor::normal({2,3});

    auto c = a - b - b;
    // AT::add_into
    a.print(6,6);
    b.print(6,6);
    c.print(6,6);
    for(size_t i = 0;i < c.getShape()[0];i++){
        for(size_t j = 0;j<c.getShape()[1];j++){
            if(c.at({i,j}) - a.at({i,j}) + 2*b.at({i,j}) > 1e-2){
                std::cout << "not match";
            }
        }
    }
    // AT::CPU::add_into_imlp(a,b,c);
    // c.print(6,6);





    // auto [g, out] = Tr::trace([&] {
    // auto x = Tr::input({2, 3}, DType::Float32, "x");
    // auto y = Tr::input({2, 3}, DType::Float32, "y");
    //     return AT::add(x, y);
    // });

}