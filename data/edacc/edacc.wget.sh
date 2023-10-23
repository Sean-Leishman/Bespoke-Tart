!#/bin/bash 

wget https://datashare.ed.ac.uk/download/DS_10283_4836.zip

unzip -t DS_10283_4836.zip
rm DS_10283_4836.zip 

mv DS_10283_4836/* . 

rm -rf D_S10283_4836
