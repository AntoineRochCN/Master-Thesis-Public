cmake_minimum_required(VERSION 3.14)
project(scrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS system json)
find_package(OpenSSL REQUIRED)

add_executable(scapper_exec scrapper.cpp)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

target_include_directories(scapper_exec PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(scapper_exec PRIVATE
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
)