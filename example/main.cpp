

// #include<iostream>

// #include <EC.hpp>
// using namespace EC;
// using namespace EC::AT;


// int main(){
//     Tensor a = Tensor::normal({3,4});
//     Tensor b = Tensor::normal({3,4});

// }

#include <vector>
#include <iostream>
#include <memory>
#include <EC.hpp>

using namespace std;
using namespace EC;
using namespace EC::AT;


struct T{};

int main(){
    AT::register_all_kernels();
    // Tensor a;
    // cout<< a.data;
    // std::shared_ptr<T> a;
    // if(!a){
    //     cout << "ok";
    // }
    Tensor a = Tensor::normal({3,4});
    // a.print();

    Tensor b = Tensor::normal({3,4});

    
    Tensor c = a * b;
    Tensor d = sin(c);
    a.print(6,6);
    b.print(6,6);
    c.print(6,6);
    d.print(6,6);
}