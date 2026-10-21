To compile scrapper, execute the following:

mkdir -p build && cd build
cmake ..
make -j

To run the scrapper:
./scrapper_exec

Be sure that the Boost lib and OpenSSL are well installed, otherwise the compilation won't work

Otherwise, one can use the python scrapper which is slightly slower and more CPU consuming, but remains effective:

python python_scrapper.py


All the collected data goes to scrapper_out.csv, so be careful to backup or clear past scraped data before launching a new collection session.
