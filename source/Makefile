CC=g++
LD=g++

INCDIR=/usr/local/include
LIBDIR=/usr/local/lib
LIBS=-lopencv_core -lopencv_videoio -lopencv_highgui \
	 -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
	 -lopencv_objdetect

smileDetection:main.o network.o
	$(LD) -o $@ $? -Wall -g -L$(LIBDIR) $(LIBS)

main.o:main.cpp
	$(CC) -c -o $@ $? -Wall -g -I$(INCDIR)

network.o:network.cpp
	$(CC) -c -o $@ $? -Wall -g -I$(INCDIR)

clean:
	rm -rf *.o smileDetection
