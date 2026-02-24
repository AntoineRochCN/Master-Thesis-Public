#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/json.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <csignal>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = net::ssl;
namespace json = boost::json;
using tcp = net::ip::tcp;

static std::ofstream outfile;
static bool keep_running = true;

void sigint_handler(int) {
    keep_running = false;
    if (outfile.is_open()) outfile.flush();
}

int main() {
    std::signal(SIGINT, sigint_handler);
    std::signal(SIGTERM, sigint_handler);

    const std::string host = "stream.binance.com";
    const std::string port = "9443";
    const std::string target = "/ws/btcusdt@depth5@100ms/btcusdt@kline_1s";

    try {
        net::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();

        tcp::resolver resolver(ioc);
        websocket::stream<ssl::stream<tcp::socket>> ws(ioc, ctx);

        auto const results = resolver.resolve(host, port);
        net::connect(ws.next_layer().next_layer(), results.begin(), results.end());
        ws.next_layer().handshake(ssl::stream_base::client);
        ws.handshake(host, target);

        std::cout << "Connected to Binance WebSocket\n";

        beast::flat_buffer buffer;
        uint64_t counter = 0;

        outfile.open("scrapper_out.csv", std::ios::app);
        if (!outfile) {
            std::cerr << "Failed to open scrapper_out.csv for writing\n";
            return 1;
        }

        outfile.seekp(0, std::ios::end);
        if (outfile.tellp() == 0) {
            outfile << "counter,timestamp,data_type";
            for (int i = 0; i < 47; ++i) outfile << ",v" << i;
            outfile << "\n";
            outfile.flush();
        }

        while (keep_running) {
            buffer.clear();
            ws.read(buffer);

            std::string_view sv{
                static_cast<const char*>(buffer.data().data()),
                buffer.size()
            };

            try {
                json::stream_parser parser;
                parser.write(sv.data(), sv.size());
                parser.finish();
                json::value jv = parser.release();

                if (!jv.is_object()) {
                    continue;
                }

                auto obj = jv.as_object();

                std::vector<double> vec(47, 0.0f);
                float data_type = 0.0f;
                uint64_t timestamp = 0;

                if (obj.contains("bids")) {
                    // depth: bids 'bids' and asks 'asks', but depends on the data flow you're looking
                    // (might be "a" and "b")
                    int pos = 0;
                    for (auto const& entry : obj["bids"].as_array()) {
                        auto level = entry.as_array();
                        if (pos < 47) vec[pos] = std::stof(level[0].as_string().c_str());
                        if (pos + 1 < 47) vec[pos + 1] = std::stof(level[1].as_string().c_str());
                        pos += 2;
                    }
                    for (auto const& entry : obj["asks"].as_array()) {
                        auto level = entry.as_array();
                        if (pos < 47) vec[pos] = std::stof(level[0].as_string().c_str());
                        if (pos + 1 < 47) vec[pos + 1] = std::stof(level[1].as_string().c_str());
                        pos += 2;
                    }
                    data_type = 0.0f;
                } else if (obj.contains("k")) {
                    auto kline = obj["k"].as_object();
                    vec[40] = std::stod(kline["o"].as_string().c_str());
                    vec[41] = std::stod(kline["h"].as_string().c_str());
                    vec[42] = std::stod(kline["l"].as_string().c_str());
                    vec[43] = std::stod(kline["c"].as_string().c_str());
                    vec[44] = std::stod(kline["v"].as_string().c_str());
                    vec[45] = std::stod(kline["Q"].as_string().c_str());
                    vec[46] = static_cast<float>(kline["n"].as_int64());
                    data_type = 1.0f;
                }

                if (obj.contains("E")) timestamp = static_cast<uint64_t>(obj["E"].as_int64());

                ++counter;
                outfile << counter << "," << timestamp << "," << std::fixed << std::setprecision(1) << data_type;
                outfile << std::setprecision(12);
                for (int i = 0; i < 47; ++i) {
                    outfile << "," << vec[i];
                }
                outfile << "\n";

                // flush periodically (every 100 messages) to reduce IO overhead
                if ((counter % 100) == 0) outfile.flush();

            } catch (const std::exception &e) {
                std::cerr << "JSON parse error or other: " << e.what() << "\n";
                continue;
            }
        }

        if (outfile.is_open()) {
            outfile.flush();
            outfile.close();
        }

    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        if (outfile.is_open()) outfile.close();
        return 1;
    }

    return 0;
}