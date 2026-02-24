To compile scrapper, execute the following:

cmake .
make scrapper_exec

To run the scrapper:
./scrapper_exec

Be sure that the Boost lib is well installed, otherwise the compilation won't work

Otherwise, one can use the python scrapper which is slightly slower and more CPU consumming, but remains effective:

python python_scrapper.python


All the collected data goes to scrapper_out.csv, so be careful to take away the past scrapped data before launching a new collection session.
