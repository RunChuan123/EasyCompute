#pragma once
#include <random>
#include <chrono>
#include <mutex>
#include <algorithm>


class RandomGenerator{
private:
    std::mt19937 gen;
    std::mutex mtx;
    RandomGenerator(){
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        gen.seed(seed);
    }
public:
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;

    static inline RandomGenerator& getInstance(){
        static RandomGenerator instance;
        return instance;
    }
    template<typename T>
    inline T getUniform(T min,T max){
        std::lock_guard<std::mutex> lock(mtx);
        std::uniform_real_distribution<T> dist(min,max);
        return dist(gen);
    }
    template<typename T>
    inline T getNormal(T mean,T stdvar){
        std::lock_guard<std::mutex> lock(mtx);
        std::normal_distribution<T> dist(mean,stdvar);
        return dist(gen);
    }
    template<typename T>
    inline void filltUniformBatch(T* data, size_t len,T min,T max){

        std::lock_guard<std::mutex> lock(mtx);
        std::uniform_real_distribution<T> dist(min,max);
        std::generate(data,data+len,[&](){
            return dist(gen);
        });
    }
    template<typename T>
    inline void fillNormalBatch(T* data, size_t len,T mean,T stdvar){

        std::lock_guard<std::mutex> lock(mtx);
        std::normal_distribution<T> dist(mean,stdvar);
        std::generate(data,data+len,[&](){
            return dist(gen);
        });
    //     std::cerr << "fill begin data=" << (void*)data << " len=" << len << "\n";
    // for (size_t i = 0; i < len; ++i) {
    //     data[i] = dist(gen);          // 如果这里崩，基本就是 data 指针不可写/堆已坏
    // }
    // std::cerr << "fill end\n";
    }


};