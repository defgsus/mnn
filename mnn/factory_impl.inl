/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

template <typename Float>
Layer<Float>* Factory<Float>::createLayer(const std::string& id, const std::string& act)
{
#define MNN__CREATE(act__) \
    if (act == Activation::act__::static_name()) \
        return createLayer<Activation::act__>(id);

    MNN__CREATE(LinearRectified);
    MNN__CREATE(Threshold);
    MNN__CREATE(ThresholdSigned);
    MNN__CREATE(Logistic);
    MNN__CREATE(Logistic10);
    MNN__CREATE(LogisticSymmetric);
    MNN__CREATE(Sine);
    MNN__CREATE(Cosine);
    MNN__CREATE(Tanh);
    MNN__CREATE(Smooth);
    MNN__CREATE(Smooth2);

#undef MNN__CREATE

    // default to Linear
    return createLayer<Activation::Linear>(id);
}


template <typename Float>
template <class ActFunc>
Layer<Float>* Factory<Float>::createLayer(const std::string& id)
{
    if (id == FeedForward<Float, ActFunc>::static_id())
        return new FeedForward<Float, ActFunc>(1, 1);

    if (id == Convolution<Float, ActFunc>::static_id())
        return new Convolution<Float, ActFunc>(1, 1, 1, 1);

    if (id == Rbm<Float, ActFunc>::static_id())
        return new Rbm<Float, ActFunc>(1, 1);

    if (id == StackSerial<Float>::static_id())
        return new StackSerial<Float>();

    if (id == StackParallel<Float>::static_id())
        return new StackParallel<Float>();

    return 0;
}


template <typename Float>
Layer<Float>* Factory<Float>::loadTextFile(const std::string& filename)
{
    std::fstream fs;
    fs.open(filename, std::ios_base::in);
    if (!fs.is_open())
        MNN_EXCEPTION("Could not open file for reading '" << filename << "'");

    return createFromStream_(fs);
}

template <typename Float>
Layer<Float>* Factory<Float>::createFromStream_(std::istream& fs)
{
    // store position
    size_t pos = fs.tellg();

    // read layer id and activation
    // (if layer does not have activation it's ignored)
    std::string id, act;
    fs >> id >> act;

    // return to start of layer description
    fs.seekg(pos, std::ios_base::beg);

    // create the required layer
    auto layer = createLayer(id, act);
    if (!layer)
        MNN_EXCEPTION("Could not create layer '" << id << "' for deserialization");

    // handle special case of stacks
    if (id == StackSerial<Float>::static_id()
     || id == StackParallel<Float>::static_id())
    {
        // id
        fs >> id;
        // version
        int ver;
        fs >> ver;
        if (ver > 1)
            MNN_EXCEPTION("Wrong version (" << ver << ") in '" << id << "'");
        // num layers
        size_t num;
        fs >> num;

        // each layer
        for (size_t i = 0; i < num; ++i)
        {
            auto sub = createFromStream_(fs);
            // XXX TODO could use an interface here
            if (auto stack = dynamic_cast<StackSerial<Float>*>(layer))
                stack->add(sub);
            else
            if (auto stack = dynamic_cast<StackParallel<Float>*>(layer))
                stack->add(sub);
            else
            {
                delete sub;
                MNN_EXCEPTION("Wrong stack object '" << layer->id() << "'");
            }
        }

        return layer;
    }

    // return to start of layer description
    fs.seekg(pos, std::ios_base::beg);

    // deserialize using layer's implementation
    layer->deserialize(fs);

    return layer;
}
