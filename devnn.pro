#-------------------------------------------------
#
# Project created by QtCreator 2014-05-09T22:58:26
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = devnn
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

QMAKE_CXXFLAGS += --std=c++11

SOURCES += main.cpp \
    trainposition.cpp

HEADERS += \
    mnn/activation.h  \
    mnn/function.h  \
    mnn/layer.h  \
    mnn/mnn.h  \
    mnn/perceptron.h  \
    mnn/perceptron_impl.inl  \
    mnn/serial.h  \
    mnn/serial_impl.inl  \
    trainposition.h
