SOURCES += \
    mnn/stack_serial_impl.inl \
    mnn/rbm_impl.inl \
    mnn/convolution_impl.inl \
    mnn/stack_parallel_impl.inl \
    mnn/feedforward_impl.inl \
    $$PWD/factory_impl.inl \
    $$PWD/stack_impl.inl \
    $$PWD/stack_split_impl.inl

HEADERS += \
    mnn/activation.h \
    mnn/function.h \
    mnn/layer.h \
    mnn/mnn.h \
    mnn/stack_serial.h \
    mnn/rbm.h \
    mnn/exception.h \
    mnn/convolution.h \
    mnn/stack_parallel.h \
    mnn/interface.h \
    mnn/feedforward.h \
    $$PWD/factory.h \
    $$PWD/stack.h \
    $$PWD/refcounted.h \
    $$PWD/stack_split.h
