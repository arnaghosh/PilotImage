all: bound

charRead: charRead.cpp
	g++ -g $^ -o $@ -Wall `pkg-config opencv --cflags --libs`

sauvola: sauvola.cpp
	g++ -g $^ -o $@ -Wall `pkg-config opencv --cflags --libs`
	
bound: 	boundBlock.cpp
	g++ -g $^ -o $@ -Wall `pkg-config opencv --cflags --libs`
	
tessCV:	 tesscv.cpp
	g++ -g $^ -o $@ -Wall -llept -ltesseract `pkg-config opencv --cflags --libs`

skew: skew.cpp
	g++ -g $^ -o $@ -Wall -llept -ltesseract `pkg-config opencv --cflags --libs`

