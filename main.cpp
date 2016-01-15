#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "mnn/mnn.h"
//#include "trainposition.h"
#include "trainmnist.h"
#include "mnistset.h"
#include "printstate.h"
#include "generate_input.h"

#define LOG(arg__) { std::cout << arg__ << std::endl; }




/** Brainwash and train a network with @p nrSamples of
    @p nrIn floats each and @p nrSamples expected results.
    Returns average error over each input sample
    and number of epochs until average error fell below 0.01 */
template <typename Float, class Net>
std::pair<Float, size_t>
      simple_test(Net& net, size_t nrIn, size_t nrSamples,
                 std::vector<Float>& input, std::vector<Float>& result,
                 bool doPrint = true)
{
	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
    if (doPrint)
    {
        net.info();
        net.dump();
        std::cout << "training some pattern with " << nrIn << " inputs...\n";
    }

	// stochastic gradient descent
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&input[nr*nrIn], &output);
		// compare
		e = result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
        if (doPrint)
        {
            std::cout << "av. error " << error << " out " << output << "           \r";
            //net.dump();
        }
        ++count;
	}
    if (doPrint)
    {
        std::cout << "took "<<count<<" epochs                                      \n";

        std::cout << "testing...\n";
    }
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&input[i*nrIn], &output);
		// compare
		e = fabs(result[i] - output);
		error += e;
        if (doPrint)
        {
            for (size_t j=0;j<nrIn;j++)
                std::cout << input[i*nrIn+j] << ", ";
            std::cout << " =\t" << output << " (abs. error " << e << ")\n";
        }
	}
	error /= nrSamples;
    if (doPrint)
        std::cout << "average abs. error " << error << "\n";
    return std::pair<Float, size_t>(error, count);

}

template <typename Float, class Net>
void simple_test_mean(Net& net, size_t nrIn, size_t nrSamples,
                 std::vector<Float>& input, std::vector<Float>& result)
{
    auto tup = simple_test(net, nrIn, nrSamples, input, result),
         av = tup, mi = tup, ma = tup;
    int num = 800;
    std::cout << "testing for " << num << " runs" << std::endl;
    for (int i = 0; i < num; ++i)
    {
        tup = simple_test(net, nrIn, nrSamples, input, result, false);
        av = std::pair<Float, size_t>(av.first + tup.first, av.second + tup.second);
        mi = std::pair<Float, size_t>(std::min(mi.first, tup.first),
                                      std::min(mi.second, tup.second));
        ma = std::pair<Float, size_t>(std::max(ma.first, tup.first),
                                      std::max(ma.second, tup.second));
    }
    av.first /= num;
    av.second /= num;
    std::cout << "epochs/error over " << num << " runs:"
              << "\naverage: " << av.second << "\t" << av.first
              << "\nmin:     " << mi.second << "\t" << mi.first
              << "\nmax:     " << ma.second << "\t" << ma.first
              << std::endl;
}


template <typename Float, class Net>
void test_some_pattern(Net& net)
{
	const size_t nrIn = 3;
	const size_t nrSamples = 5;
	std::vector<Float> input(
	{ 	0.0, 0.0, 0.0,
		0.2, 0.3, 0.2,
		0.7, 0.6, 0.5,
		0.9, 0.8, 0.9,
		0.1, 0.5, 0.9
	} );
	std::vector<Float> result(
	{
		0.0,
		0.23,
		0.6,
		0.86,
		0.5
	});
    simple_test_mean(net, nrIn, nrSamples, input, result);
};

template <typename Float, class Net>
void test_xor_pattern(Net& net)
{
    const size_t nrIn = 2;
	const size_t nrSamples = 4;
	std::vector<Float> input(
    { 	0.1, 0.1,
        0.1, 0.9,
        0.9, 0.1,
        0.9, 0.9
	} );
	std::vector<Float> result(
	{
		0.1,
		0.9,
		0.9,
		0.1
	});
    simple_test_mean(net, nrIn, nrSamples, input, result);
};







template <typename Float, class Net>
void xor_test(Net& net)
{

	const size_t nrIn = 2;
	const size_t nrSamples = 4;
	const Float X = 0.9;
	const Float O = 0.1;
	std::vector<Float> xor_input(
	{
		O, O,
		O, X,
		X, O,
		X, X
	});
	std::vector<Float> xor_result(nrSamples);

	for (size_t i=0;i<nrSamples;i++)
	{
		size_t k = (xor_input[i*nrIn]>0.5);
		for (size_t j=1;j<nrIn;j++)
			k = k xor (xor_input[i*nrIn+j]>0.5);
		xor_result[i] = (k)? X : O;
	}

	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
	net.info();
	net.dump();

	// stochastic gradient descent
	std::cout << "training "<<nrIn<<"-XOR pattern...\n";
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&xor_input[nr*nrIn], &output);
		// compare
		e = xor_result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
		std::cout << "av. error " << error << " out " << output << "           \r";
		//net.print();
		count++;
	}
	std::cout << "took "<<count<<" epochs                                      \n";

	std::cout << "testing...\n";
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&xor_input[i*nrIn], &output);
		// compare
		e = fabs(xor_result[i] - output);
		error += e;
		for (size_t j=0;j<nrIn;j++)
			std::cout << xor_input[i*nrIn+j] << ", ";
		std::cout << " =\t" << output << " (abs. error " << e << ")\n";
	}
	error /= nrSamples;
	std::cout << "average abs. error " << error << "\n";
}





/* ------------ simple pattern learner -----------
   using above functions
   averaged over a number of runs
   to quickly evaluate networks, and learning and activation functions
   */

template <typename F>
void maint()
{
    //MNN::Perceptron<float, MNN::Activation::Linear> net;

    MNN::StackSerial<F> net;
#if 0
    // about 540 epochs for av error < 0.01
    auto l1 = new MNN::Perceptron<F, MNN::Activation::Tanh>(10,10, 1);
    auto l2 = new MNN::Perceptron<F, MNN::Activation::Logistic>(10,10, 1);
    auto l3 = new MNN::Perceptron<F, MNN::Activation::Linear>(10,10, 0.1);

    l1->setMomentum(.5);
    //l1->setLearnRateBias(0.01);
    l2->setMomentum(.5);
    l3->setMomentum(.5);
#else
    F learnr = 1.;
    auto l1 = new MNN::Perceptron<F, MNN::Activation::Tanh>(10,40, learnr * 1);
    auto l2 = new MNN::Perceptron<F, MNN::Activation::Logistic>(40,10, learnr * 1);
    auto l3 = new MNN::Perceptron<F, MNN::Activation::Linear>(10,10, learnr * 0.1);
    l1->setMomentum(.5);
    l2->setMomentum(.5);
    l3->setMomentum(.5);
#endif

    net.add(l1);
    net.add(l2);
    net.add(l3);

    test_xor_pattern<F>(net);
}






/*    / \    */
/*   /   \   */
/*  /.....\  */
/*     .     */
/*           */
template <typename Float, class Rbm>
class RbmPyramid
{
public:
    struct Sample
    {
        Sample() : err_cd(0.), err_rec(0.) { }
        std::vector<Float> data;
        Float err_cd, err_rec;
    };

private:
    std::vector<size_t> numCells_;
    std::vector<Rbm*> rbm_;
    MNN::StackSerial<Float> stack_;
    std::vector<Sample*> samples_, higherSamples_;
    const size_t cdSteps_ = 4;
    const Float learnRate_ = 0.05;
    const Float momentum_ = .7;
public:

    void clearSamples() { for (auto s : samples_) delete s; samples_.clear(); }
    void clearHigherSamples() { for (auto s : higherSamples_) delete s; higherSamples_.clear(); }

    size_t numIn() const { return numCells_.empty() ? 0 : numCells_[0]; }

    void setSize(const std::vector<size_t>& numCells)
    {
        numCells_ = numCells;
        for (size_t i=1; i<numCells.size(); ++i)
        {
            auto rbm = new Rbm(numCells[i-1], numCells[i]);
            rbm->brainwash();
            rbm->setMomentum(momentum_);
            rbm_.push_back(rbm);
            stack_.add(rbm);
        }
    }

    void addSample(const Float* s)
    {
        auto sam = new Sample;
        sam->data.resize(numIn());
        for (auto& f : sam->data)
            f = *s++;
        samples_.push_back(sam);
    }

    std::string layerFilename(size_t index) const
    {
        std::stringstream str;
        str << "rbm_layer_" << index << ".txt";
        return str.str();
    }

    void loadLayer(size_t index)
    {
        rbm_[index]->loadTextFile(layerFilename(index));
        std::cout << "loaded rbm layer " << index << " ("
                  << layerFilename(index) << ")\n";
        rbm_[index]->info();
    }

    /** Creates the net output for each sample_ into higherSamples_ */
    void createHigherSamples(size_t index)
    {
        LOG("creating output for each sample for layer " << index);
        clearHigherSamples();
        if (index == 0)
        {
            for (size_t i = 0; i < samples_.size(); ++i)
            {
                // get space for output
                auto hsam = new Sample;
                higherSamples_.push_back(hsam);
                hsam->data.resize(rbm_[index]->numOut());
                // prop first layer
                rbm_[0]->fprop(&samples_[i]->data[0], &hsam->data[0]);
            }
        }
        else
        {
            std::vector<Float> output(stack_.numOut());
            for (size_t i = 0; i < samples_.size(); ++i)
            {
                // get space for output
                auto hsam = new Sample;
                higherSamples_.push_back(hsam);
                hsam->data.resize(rbm_[index]->numOut());

                // prop whole stack
                stack_.fprop(&samples_[i]->data[0], &output[0]);
                // get output from specific layer
                for (size_t j=0; j<hsam->data.size(); ++j)
                    hsam->data[j] = rbm_[index]->output(j);
            }
        }
    }

    void trainLayer(size_t index, size_t maxEpoch = 300000)
    {
        LOG("\n------ TRAIN LAYER #" << index << " ------");
        Float err_min = -1.,
              err_max = 0.,
              err_sum = 0.,
              last_save_err = -1.;
        Rbm* rbm = rbm_[index];
        size_t epoch = 0, err_count = 0;
        while (epoch++ <= maxEpoch)
        {
            // -- choose sample --

            Sample * sample;
            // real-data -> first layer
            if (index == 0)
            {
                size_t samIndex = size_t(rand()) % samples_.size();
                sample = samples_[samIndex];
            }
            // layer n-1 -> n
            else
            {
                size_t samIndex = size_t(rand()) % higherSamples_.size();
                sample = higherSamples_[samIndex];
            }

            // contrastive divergance training
            Float err = rbm->contrastiveDivergence(&sample->data[0], cdSteps_, learnRate_);
            //err = err / (rbm->numOut() * rbm->numIn()) * 100;
            sample->err_cd = err;

            // gather error stats
            if (err_min < 0.)
                err_min = err;
            else
                err_min = std::min(err_min, err);
            err_max = std::max(err_max, err);

            // CD error of 0. means probably no state at all
            if (err > 0.)
            {
                err_sum += err;
                ++err_count;
            }

            if (epoch == 1 || (epoch % 1000) == 0)
            {
                LOG(         "step " << std::left << std::setw(9) << epoch
                          << std::setprecision(3)
                          << " err " << std::setw(9) << err_min
                          << " - " << std::setw(9) << err_max
                          << " av " << std::setw(9) << (err_sum / err_count)
                          << " avweight " << rbm->getWeightAverage()
                   );

                if (err_max < .4 &&
                        (last_save_err < 0. || err_max < last_save_err))
                {
                    last_save_err = err_max;
                    rbm->saveTextFile(layerFilename(index));
                    LOG("saved layer #" << index << " as '" << layerFilename(index) << "'");
                }

                if (err_max < 0.1)
                    break;

                err_max = 0.;
                err_min = -1.;
                err_sum = 0.;
                err_count = 0;
            }
        }
    }
};



void trainRbmPyramid()
{
    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/train-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/train-images.idx3-ubyte");
    size_t numIn = set.width() * set.height();

    // init pyramid
    RbmPyramid<float, MNN::Rbm<float, MNN::Activation::Logistic>> stack;
    stack.setSize({ numIn, 300, 200, 10 });

    // copy samples
    for (size_t i=0; i<set.numSamples(); ++i)
        stack.addSample(set.image(i));

#if 0
    for (int i=0; i<3; ++i)
    {
        stack.trainLayer(i);
        stack.createHigherSamples(i);
    }
#else
    stack.loadLayer(0);
    //stack.loadLayer(1);
    stack.createHigherSamples(0);
    stack.trainLayer(2);
#endif
}

void testRbmPyramid()
{
    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");
    //size_t numIn = set.width() * set.height();

}




template <typename Net>
void evaluateRbm(Net rbm, const MnistSet& set)
{
    struct Sample
    {
        size_t index;
        float err_cd, err_rec;
        std::string img;
    };

    std::vector<Sample> samples;
    for (size_t i = 0; i < set.numSamples(); ++i)
    {
        Sample s;
        s.index = i;
        s.err_cd = rbm->cd(set.image(i), 3, 0.);
        s.err_rec = rbm->compareInput(set.image(i));
        std::stringstream strs;
        printStateAscii(rbm->input(), set.width(), set.height(), strs);
        s.img = strs.str();

        samples.push_back(s);
    }

    std::sort(samples.begin(), samples.end(),
              [](const Sample& l, const Sample& r)
    {
        return l.err_rec < r.err_rec;
        //return l.err_cd < r.err_cd;
    });

    for (const auto& s : samples)
    {
        std::cout << "image #" << s.index
                  << ", cd " << s.err_cd
                  << ", rec " << s.err_rec
                  << "\n" << s.img;
        //printState(rbm->output(), rbm->numOut(), 1);
        std::cout << std::endl;

        std::cin.get();
    }
}

template <typename Float, class Rbm>
void runInputApproximationRbm(Rbm& net)
{
    assert(net.numIn() == 28*28);

    GenerateInput<Float> gen(.5, .9);
    gen.initializeInput();

    const size_t numIt = 1000;
    size_t it = 0;
    while (true)
    {
        gen.approximateInputRbm(net, numIt, 1);
        it += numIt;

        std::cout << "iteration " << it << ", error best " << gen.error()
                  << ", worst " << gen.errorWorst() << "\n";
        printStateAscii(gen.input(), 28, 28);
        std::cout << std::endl;
    }
}


template <typename F>
void testRbm()
{
#define MNIST

#ifdef MNIST
    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");
    size_t numIn = set.width() * set.height();

//    printStateAscii(set.image(0), set.width(), set.height());
#else
    size_t numIn = 10;
    std::vector<F> input(numIn);
    for (auto& f : input)
        f = MNN::rnd(F(0), F(1));
#endif

    auto rbm = new MNN::Rbm<F, MNN::Activation::Logistic>(numIn, 100, 1);
    rbm->brainwash();
    rbm->setMomentum(0.5);

#if 0
    // load previous
    rbm->loadTextFile("mnist_rbm_20h.txt");
    //rbm->dump();

    evaluateRbm(rbm, set);
    return;
#endif

    rbm->info();

    F err = 0., err_sum = 0., err_min = 0., err_max = 0.;
    size_t err_count = 1;
    const size_t num = 250000;
    for (size_t it=0; it<num; ++it)
    {
        if (it % 1000 == 0)
        {
            F av_err = (err_sum / err_count);
            //rbm->dump();
            std::cout << "step " << std::left << std::setw(9) << it
                      << " err " << std::setw(9) << err_min
                      << " - " << std::setw(9) << err_max
                      << " av " << std::setw(9) << av_err
                      << " avweight " << rbm->getWeightAverage()
                      << std::endl;

            if (av_err < 4. && it > 60000)
                runInputApproximationRbm<F>(*rbm);

            err_max = 0.;
            err_min = -1.;
            err_sum = 0;
            err_count = 0;
        }

#ifdef MNIST
        size_t idx = rand() % set.numSamples();
        //idx = idx % 10;
        err = rbm->contrastiveDivergence(set.image(idx), 2, 0.0005);
        err = 100 * err / (rbm->numOut() * rbm->numIn());
        //err = rbm->compareInput(set.image(idx));
#else
        err = rbm->cd(&input[0], 2, 0.1);
#endif
        // gather error stats
        if (err > 0.)
        {
            if (err_min < 0.)
                err_min = err;
            else
                err_min = std::min(err_min, err);
            err_max = std::max(err_max, err);

            err_sum += err;
            ++err_count;
        }

        if (err > 0.0 && err < 20.)
        {
            //printStateAscii(rbm->inputs(), set.width(), set.height());
            //printStateAscii(rbm->outputs(), rbm->numOut(), 1);
        }
    }

#if 0
    rbm->saveTextFile("mnist_rbm_20h.txt");
#endif

}


// Simple test for auto-encoding / reconstruction error
template <typename Float>
void trainRecon()
{
    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/train-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/train-images.idx3-ubyte");
    set.scale(14, 14);
    size_t numIn = set.width() * set.height();

    auto net = new MNN::Perceptron<Float, MNN::Activation::Linear>(
                numIn, numIn/2, 1, false);
    net->setMomentum(.9);
    net->brainwash(0.1);
    //MNN::initPassThrough(net);
    Float learnRate = 0.0002;

    net->info();

    size_t epoch = 0, err_count = 0;
    Float err_sum = 0., err_min = -1., err_max = 0.,
          lastWeights = 0.;
    while (true)
    {
        uint32_t index = uint32_t(rand()) % set.numSamples();
        const Float* image = set.image(index);
        const Float* noise_image
                = set.getNoisyImage(index, MNN::rnd(-0.3,.0), MNN::rnd(0.,.3));

        Float error = 100. * net->reconstructionTraining(noise_image, image, learnRate);
        ++epoch;

        err_sum += error;
        if (err_min < 0. || error < err_min)
            err_min = error;
        err_max = std::max(err_max, error);
        ++err_count;

        if (epoch % 5000 == 0)
        {
            Float weights = net->getWeightAverage();
            LOG("epoch " << std::left << std::setw(8) << epoch
                << " error " << std::setw(9) << err_min
                << " - " << std::setw(9) << err_max
                << " av " << std::setw(9) << (err_sum / err_count)
                << " weights " << std::setw(9) << weights
                << " inc " << std::setw(9) << ((weights - lastWeights) / learnRate)
                );

            lastWeights = weights;
            err_sum = err_max = 0.;
            err_min = -1.;
            err_count = 0;

            //printStateAscii(net->weights(), set.width(), set.height(), 8.f);
#if 0
            std::vector<Float> recon(numIn);
            net->reconstruct(image, &recon[0]);
            printStateAscii(&recon[0], set.width(), set.height());
#endif
        }
    }
}



template <typename Float>
void trainAutoencoderStack()
{
    const Float sizeScale = .75;

    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/train-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/train-images.idx3-ubyte");
    set.normalize();
    set.scale(14, 14);
    size_t numIn = set.width() * set.height();

    auto net = new MNN::StackSerial<Float>();

    auto layer = new MNN::Perceptron<Float, MNN::Activation::Linear>(
                    numIn, numIn * 2    , 1, false);
    layer->setMomentum(.9);
    layer->brainwash(0.1);
    Float learnRate = 0.0009;

    layer->info();

    std::vector<Float> buffer;

    size_t epoch = 0, err_count = 0;
    Float err_sum = 0., err_min = -1., err_max = 0.,
          lastWeights = 0.;
    while (true)
    {
        // get training sample
        uint32_t index = uint32_t(rand()) % set.numSamples();
        const Float* image = set.image(index);

        // propagate through already trained stack
        if (net->numLayer())
        {
            if (buffer.size() != net->numOut())
                buffer.resize(net->numOut());
            net->fprop(image, &buffer[0]);
            image = &buffer[0];
        }

        Float error = 100. * layer->reconstructionTraining(image, learnRate);
        ++epoch;

        err_sum += error;
        if (err_min < 0. || error < err_min)
            err_min = error;
        err_max = std::max(err_max, error);
        ++err_count;

        if (epoch % 5000 == 0)
        {
            Float weightAv = layer->getWeightAverage(),
                  weightInc = weightAv - lastWeights;
            LOG("epoch " << std::left << std::setw(8) << epoch
                << " error " << std::setw(9) << err_min
                << " - " << std::setw(9) << err_max
                << " av " << std::setw(9) << (err_sum / err_count)
                << " weights " << std::setw(9) << weightAv
                << " inc " << std::setw(9) << (weightInc / learnRate)
                );

            lastWeights = weightAv;
            err_sum = err_max = 0.;
            err_min = -1.;
            err_count = 0;

            // finished training this layer?
            if (weightInc / learnRate < 0.01
                && epoch > 120000)
            {
                net->add(layer);
                net->saveTextFile("../autoencoder-stack-mnist.txt");

                size_t newOut = net->numOut() * sizeScale;
                if (newOut < 100)
                    break;

                layer = new MNN::Perceptron<Float, MNN::Activation::Linear>(
                            net->numOut(), newOut, 1, false);
                layer->setMomentum(.9);
                layer->brainwash(0.1);

                layer->info();

                epoch = 0;
            }
        }
    }
}




int main()
{
	srand(time(NULL));

    //TrainPosition t; t.exec(); return 0;
    TrainMnist t; t.exec(); return 0;

    //maint<double>();
    //testRbm<float>();
    //trainRbmPyramid();

    trainAutoencoderStack<float>();

	return 0;
}
