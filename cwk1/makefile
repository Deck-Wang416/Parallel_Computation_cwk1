#
# Simple makefile for Coursework 1. Normaly you would not upload a makefile
# for this assignment, but if you do, ensure (a) the executable name is unchanged,
# and (b) it works on the Gradescope autograder (in particular, do not change the
# name of the C-compiler to something that only exists on your system).
#
EXE = cwk1
CC = gcc
CCFLAGS = -fopenmp -lm
GRAPHICS = NONE

OS = $(shell uname)

ifeq ($(OS), Linux)
	CCFLAGS += -lX11 -lGL -lglfw
endif

ifeq ($(OS), Darwin)
	CCFLAGS += -lglfw -framework OpenGL -L /opt/homebrew/lib -I /opt/homebrew/include
endif

all:
	@echo For graphical output, you may need to modify the makefile to match your installation of glfw.
	@echo
	$(CC) $(CCFLAGS) -o $(EXE) -D$(GRAPHICS) cwk1.c

