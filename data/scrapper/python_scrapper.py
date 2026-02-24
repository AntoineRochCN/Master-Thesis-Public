import asyncio
import json
import signal
import websockets

keep_running = True

def sigint_handler(sig, frame):
    global keep_running
    print("\nStopping the scrapper...")
    keep_running = False

async def main():
    global keep_running
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    host = "stream.binance.com"
    port = 9443
    target = "/ws/btcusdt@depth5@100ms/btcusdt@kline_1s"
    uri = f"wss://{host}:{port}{target}"

    output_path = "scrapper_out.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter = 0

    try:
        async with websockets.connect(uri, ssl=True) as ws:
            print(f"Connected to Binance WebSocket")

            file_exists = output_path.exists()
            
            with open(output_path, "a", encoding="utf-8") as f:
                if not file_exists or output_path.stat().st_size == 0:
                    header = "counter,timestamp,data_type," + ",".join([f"v{i}" for i in range(47)]) + "\n"
                    f.write(header)
                
                while keep_running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        vec = [0.0] * 47
                        data_type = 0.0
                        timestamp = data.get("E", 0)

                        if "bids" in data:
                            pos = 0
                            for level in data["bids"]:
                                if pos < 47: vec[pos] = float(level[0])
                                if pos + 1 < 47: vec[pos + 1] = float(level[1])
                                pos += 2
                            
                            for level in data["asks"]:
                                if pos < 47: vec[pos] = float(level[0])
                                if pos + 1 < 47: vec[pos + 1] = float(level[1])
                                pos += 2
                            data_type = 0.0
                        
                        elif "k" in data:
                            k = data["k"]
                            vec[40] = float(k["o"]) # Open
                            vec[41] = float(k["h"]) # High
                            vec[42] = float(k["l"]) # Low
                            vec[43] = float(k["c"]) # Close
                            vec[44] = float(k["v"]) # Volume
                            vec[45] = float(k["Q"]) # Asset volume
                            vec[46] = float(k["n"]) # Number of trades
                            data_type = 1.0

                        counter += 1

                        vec_str = ",".join([f"{v:.12f}" if i != 46 else f"{v:.1f}" for i, v in enumerate(vec)])
                        line = f"{counter},{timestamp},{data_type:.1f},{vec_str}\n"
                        f.write(line)

                        if counter % 100 == 0:
                            f.flush()

                    except Exception as e:
                        print(f"Parsing error : {e}")
                        continue

    except Exception as e:
        print(f"Connexion error : {e}")

if __name__ == "__main__":
    asyncio.run(main())