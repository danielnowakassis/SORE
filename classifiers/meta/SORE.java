/*
 *    SORE.java
 *
 *    @author Heitor Murilo Gomes (heitor dot gomes at waikato dot ac dot nz)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.AutoML.RandomSearch.RandomSearch;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.*;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;



public class SORE extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train on instances.", Classifier.class, "HPO.RandomSearch.RandomSearch -f (HT.json)");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models.", 100, 1, Integer.MAX_VALUE);

    // SUBSPACE CONFIGURATION
    public MultiChoiceOption subspaceModeOption = new MultiChoiceOption("subspaceMode", 'o',
            "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
            new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
                    "Percentage (M * (m / 100))"},
            new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 3);

    public IntOption subspaceSizeOption = new IntOption("subspaceSize", 'm',
            "# attributes per subset for each classifier. Negative values = totalAttributes - #attributes", 60, Integer.MIN_VALUE, Integer.MAX_VALUE);

    // TRAINING
    public MultiChoiceOption trainingMethodOption = new MultiChoiceOption("trainingMethod", 't',
            "The training method to use: Random Patches, Random Subspaces or Bagging.",
            new String[]{"Random Subspaces", "Resampling (bagging)", "Random Patches"},
            new String[]{"RandomSubspaces", "Resampling", "RandomPatches"}, 2);

    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
            "The lambda parameter for bagging.", 6.0, 1, Float.MAX_VALUE);

    // DRIFT and WARNING DETECTION
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

    // DISABLING DRIFT DETECTION and BKG LEARNER (warning is also disabled in this case)
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
            "Should use drift detection? If disabled, then the bkg learner is also disabled.");

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
            "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);

    public IntOption randomSeedOption  = new IntOption("seedSizeSubspace", 'S',
            "Random seed used in the random subspace size.", 1, 0, Integer.MAX_VALUE);

    public FlagOption disablesubBaggingOption = new FlagOption("subBaggingOption", 'B',
            "Should use subBagging");

    public FlagOption disableRegularization = new FlagOption("disableRegularization", 'R',
            "Should use regularization");

    public FlagOption disableWeightedVoting = new FlagOption("disableWeightedVoting", 'W',
            "Should use disableWeightedVoting");

    public FlagOption disableEnsembleSelection = new FlagOption("disableEnsembleSelection", 'A',
            " disableEnsembleSelection");

    public FlagOption enableBag = new FlagOption("enableBag", 'E',
            "Bag instead of pasting in regularization");

    CopyOnWriteArrayList<Pair<Boolean, Integer>> warningsAndDrifts = new CopyOnWriteArrayList<>();

    public IntOption numberRejections  = new IntOption("numberRejections", 'r',
            "maximum number of rejections (0 = do not use number of rejections).", 5, 0, Integer.MAX_VALUE);


    public IntOption windowObservationSize  = new IntOption("windowObservationSize", 'w',
            "Size of the observation window to refine statistics and select learners in the voting.", 1000, 0, Integer.MAX_VALUE);

    public static final int TRAIN_RANDOM_SUBSPACES = 0;
    public static final int TRAIN_RESAMPLING = 1;
    public static final int TRAIN_RANDOM_PATCHES = 2;

    protected static final int FEATURES_M = 0;
    protected static final int FEATURES_SQRT = 1;
    protected static final int FEATURES_SQRT_INV = 2;
    protected static final int FEATURES_PERCENT = 3;

    protected SOREClassifier[] ensemble;
    protected long instancesSeen;
    protected ArrayList<ArrayList<Integer>> subspaces;

    //statistic window size
    protected double avgAccuracyWindowLearner;

    private ExecutorService executor;

    @Override
    public void resetLearningImpl() {
        this.instancesSeen = 0;
        this.warningsAndDrifts = new CopyOnWriteArrayList<>();
        // Multi-threading
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1)
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent.
        // this.executor will be null and not used...
        if(numberOfJobs != AdaptiveRandomForest.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if(this.ensemble == null)
            initEnsemble(instance);

        this.warningsAndDrifts.clear();

        //System.out.println(instancesSeen);
        double accWindowLearner = 0.0;

        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            double[] rawVote = this.ensemble[i].getVotesForInstance(instance);
            DoubleVector vote = new DoubleVector(rawVote);
            InstanceExample example = new InstanceExample(instance);

            int trueClass = (int) instance.classValue();
            int predictedClass = Utils.maxIndex(vote.getArrayRef());

            /*To avoid that domains with high noise incidence fully participate in the model creation.
             * We use the strategy that for every five rejections,
             * one instance is trained even though it is classified correctly*/
            boolean willTrain = (this.disableRegularization.isSet()) || trueClass != predictedClass;



            //willTrain = true;

            //hit
            if (this.disablesubBaggingOption.isSet() && !willTrain) {
                this.ensemble[i].UntrainedClasses[trueClass]++;
                if (this.ensemble[i].UntrainedClasses[trueClass] >= this.numberRejections.getValue() && this.numberRejections.getValue() > 0) {
                    this.ensemble[i].UntrainedClasses[trueClass] = 0;
                    willTrain = true;
                }
            }

            if (willTrain) {
                if(this.executor != null){
                    if(this.trainingMethodOption.getChosenIndex() == TRAIN_RANDOM_SUBSPACES) {
                        TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i],
                                instance, 1, this.instancesSeen, this.classifierRandom);
                        trainers.add(trainer);
                    }
                    else{
                        int k =  MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
                        if (k > 0){
                            double weight = k;
                            TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i],
                                    instance, weight, this.instancesSeen, this.classifierRandom);
                            trainers.add(trainer);
                        }
                    }
                }else{
                    // Train using random subspaces without resampling, i.e. all instances are used for training.
                    if(this.trainingMethodOption.getChosenIndex() == TRAIN_RANDOM_SUBSPACES) {
                        this.ensemble[i].trainOnInstance(instance,1, this.instancesSeen, null);
                    }
                    // Train using random patches or resampling, thus we simulate online bagging with poisson(lambda=...)
                    else {
                        int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
                        if (k > 0) {
                            double weight = k;
                            this.ensemble[i].trainOnInstance(instance, weight, this.instancesSeen, this.classifierRandom);
                        }
                    }

                }
            }else if (!this.disablesubBaggingOption.isSet()){
                if(this.executor != null){
                    if(this.trainingMethodOption.getChosenIndex() == TRAIN_RANDOM_SUBSPACES) {
                        TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i],
                                instance, 1, this.instancesSeen, this.classifierRandom);
                        trainers.add(trainer);
                    }
                    else{
                        int k =  MiscUtils.poisson(1, this.classifierRandom);
                        if (k > 0){
                            if(!enableBag.isSet())
                                k=1;
                            double weight = k;
                            TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i],
                                    instance, weight, this.instancesSeen, this.classifierRandom);
                            trainers.add(trainer);
                        }
                    }
                }else{
                    // Train using random subspaces without resampling, i.e. all instances are used for training.
                    if(this.trainingMethodOption.getChosenIndex() == TRAIN_RANDOM_SUBSPACES) {
                        this.ensemble[i].trainOnInstance(instance,1, this.instancesSeen, null);
                    }
                    // Train using random patches or resampling, thus we simulate online bagging with poisson(lambda=...)
                    else {
                        int k = MiscUtils.poisson(1, this.classifierRandom);
                        if (k > 0) {
                            k=1;
                            double weight = k;
                            this.ensemble[i].trainOnInstance(instance, weight, this.instancesSeen, this.classifierRandom);
                        }
                    }

                }
            }

            accWindowLearner += this.ensemble[i].accuracyWindowLearner;

            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());





        }

        if (accWindowLearner > 0.0) {
            avgAccuracyWindowLearner = (accWindowLearner / this.ensemble.length);
        }


        if(this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

        if(!this.warningsAndDrifts.isEmpty()){
            warningsAndDrifts.sort(Comparator.comparing(p -> p.second));
            //System.out.println(warningsAndDrifts);
            for(Pair p : this.warningsAndDrifts){


                this.ensemble[(int) p.second].reset(instance, this.instancesSeen, this.classifierRandom);

                //System.out.println("Warning " + this.ensemble[(int) p.second].bkgLearner + " " + (int) p.second + " " + this.instancesSeen+ " " + (boolean)p.first);
            }
        }


    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        testInstance.setMissing(instance.classAttribute());
        testInstance.setClassValue(0.0);
        if(this.ensemble == null)
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();
        boolean shouldVote = true;
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            if (this.windowObservationSize.getValue() > 0)
                shouldVote = this.disableEnsembleSelection.isSet() || (this.ensemble[i].accuracyWindowLearner >= avgAccuracyWindowLearner);
            if (shouldVote) {
                DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
                if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                if(!this.disableWeightedVoting.isSet()) {
                    double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                    if (acc > 0.0) {
                        for (int v = 0; v < vote.numValues(); ++v) {
                            vote.setValue(v, vote.getValue(v) * acc);
                        }
                    }
                }

                combinedVote.addValues(vote);
                }
            }
        }

        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
        cleanThreads();
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        cleanThreads();
        return null;
    }

    public void cleanThreads() {
        if(this.executor != null) {
            this.executor.shutdownNow();
            this.executor = null;
        }
    }

    protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new SOREClassifier[ensembleSize];

        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();


        // #1 Select the size of k, it depends on 2 parameters (subspaceSizeOption and subspaceModeOption).
        int k = this.subspaceSizeOption.getValue();
        if(this.trainingMethodOption.getChosenIndex() != SORE.TRAIN_RESAMPLING) {
            // PS: This applies only to subspaces and random patches option.
            int n = instance.numAttributes()-1; // Ignore the class label by subtracting 1

            switch(this.subspaceModeOption.getChosenIndex()) {
                case SORE.FEATURES_SQRT:
                    k = (int) Math.round(Math.sqrt(n)) + 1;
                    break;
                case SORE.FEATURES_SQRT_INV:
                    k = n - (int) Math.round(Math.sqrt(n) + 1);
                    break;
                case SORE.FEATURES_PERCENT:
                    double percent = k < 0 ? (100 + k)/100.0 : k / 100.0;
                    k = (int) Math.round(n * percent);

                    if(Math.round(n * percent) < 2)
                        k = (int) Math.round(n * percent) + 1;
                    break;
            }
            // k is negative, use size(features) + -k
            if(k < 0)
                k = n + k;

            // #2 generate the subspaces
            if(this.trainingMethodOption.getChosenIndex() == SORE.TRAIN_RANDOM_SUBSPACES ||
                    this.trainingMethodOption.getChosenIndex() == SORE.TRAIN_RANDOM_PATCHES) {
                if(k != 0 && k < n) {
                    // For low dimensionality it is better to avoid more than 1 classifier with the same subspaces,
                    // thus we generate all possible combinations of subsets of features and select without replacement.
                    // n is the total number of features and k is the actual size of the subspaces.
                    if(n <= 20 || k < 2) {
                        if(k == 1 && instance.numAttributes() > 2)
                            k = 2;
                        // Generate all possible combinations of size k
                        this.subspaces = SORE.allKCombinations(k, n);
                        for(int i = 0 ; this.subspaces.size() < this.ensemble.length ; ++i) {
                            i = i == this.subspaces.size() ? 0 : i;
                            ArrayList<Integer> copiedSubspace = new ArrayList<>(this.subspaces.get(i));
                            this.subspaces.add(copiedSubspace);
                        }
                    }

                    // For high dimensionality we can't generate all combinations as it is too expensive (memory).
                    // On top of that, the chance of repeating a subspace is lower, so we can just randomly generate
                    // subspaces without worrying about repetitions.
                    else {
                        this.subspaces = SORE.localRandomKCombinations(k, n,
                                this.ensembleSizeOption.getValue(), this.classifierRandom);
                    }
                }
                // k == 0 or k > n (subspace size greater than the total number of features), then default to resampling
                else {
                    this.trainingMethodOption.setChosenIndex(SORE.TRAIN_RESAMPLING);
                }
            }
        }

        // Obtain the base learner. It is not restricted to a specific learner.
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();

        if(baseLearner instanceof RandomSearch){
            ((RandomSearch) baseLearner).rsO.setValue(this.randomSeedOption.getValue());
        }

        for(int i = 0 ; i < ensembleSize ; ++i) {
            switch(this.trainingMethodOption.getChosenIndex()) {
                case SORE.TRAIN_RESAMPLING:
                    this.ensemble[i] = new SOREClassifier(
                            i,
                            baseLearner.copy(),
                            (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(),
                            this.instancesSeen,
                            this.disableDriftDetectionOption.isSet(),
                            this.driftDetectionMethodOption,
                            this.warningsAndDrifts,
                            windowObservationSize.getValue()

                    );
                    break;
                case SORE.TRAIN_RANDOM_SUBSPACES:
                case SORE.TRAIN_RANDOM_PATCHES:
                    int selectedValue = this.classifierRandom.nextInt(subspaces.size());
                    ArrayList<Integer> subsetOfFeatures = this.subspaces.get(selectedValue);
                    subsetOfFeatures.add(instance.classIndex());
                    this.ensemble[i] = new SOREClassifier(
                            i,
                            baseLearner.copy(),
                            (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(),
                            this.instancesSeen,
                            this.disableDriftDetectionOption.isSet(),
                            this.driftDetectionMethodOption,
                            subsetOfFeatures,
                            instance,
                            warningsAndDrifts,
                            windowObservationSize.getValue()
                    );
                    this.subspaces.remove(selectedValue);
                    break;
            }
            this.ensemble[i].UntrainedClasses = new long[instance.numClasses()];
        }

    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == SORE.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    @Override
    public Classifier[] getSublearners() {
        /* Extracts the reference to the base learner object from within the ensemble of SOREClassifier */
        Classifier[] baseModels = new Classifier[this.ensemble.length];
        for(int i = 0 ; i < baseModels.length ; ++i)
            baseModels[i] = this.ensemble[i].classifier;
        return baseModels;
    }

    public static ArrayList<ArrayList<Integer>> localRandomKCombinations(int k, int length,
                                                                         int nCombinations, Random random) {
        ArrayList<ArrayList<Integer>> combinations = new ArrayList<>();
        for(int i = 0 ; i < nCombinations ; ++i) {
            ArrayList<Integer> combination = new ArrayList<>();
            // Add all possible items
            for(int j = 0 ; j < length ; ++j)
                combination.add(j);
            // Randomly remove each item by index using the current size
            // Out of "length" items, maintain only "k" items.
            for(int j = 0 ; j < (length - k) ; ++j)
                combination.remove(random.nextInt(combination.size()));

            combinations.add(combination);
        }
        return combinations;
    }

    private static void allKCombinationsInner(int offset, int k, ArrayList<Integer> combination, long originalSize,
                                              ArrayList<ArrayList<Integer>> combinations) {
        if (k == 0) {
            combinations.add(new ArrayList<>(combination));
            return;
        }
        for (int i = offset; i <= originalSize - k; ++i) {
            combination.add(i);
            allKCombinationsInner(i+1, k-1, combination, originalSize, combinations);
            combination.remove(combination.size()-1);
        }
    }

    public static ArrayList<ArrayList<Integer>> allKCombinations(int k, int length) {
        ArrayList<ArrayList<Integer>> combinations = new ArrayList<>();
        ArrayList<Integer> combination = new ArrayList<>();
        allKCombinationsInner(0, k, combination, length, combinations);
        return combinations;
    }

    // Inner class representing the base learner of SRP.
    protected class SOREClassifier {
        public int indexOriginal;
        public long createdOn;
        public Classifier classifier;

        protected long[] UntrainedClasses;

        // Stores current model subspace representation of the original instances.
        public Instances subset;
        public int[] featureIndexes;

        // Drift detection
        public boolean disableDriftDetector;

        protected ChangeDetector driftDetectionMethod;
        // The drift and warning object parameters.
        protected ClassOption driftOption;

        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        public int numberOfDriftsDetected;

        CopyOnWriteArrayList<Pair<Boolean, Integer>> warningsAndDrifts;

        // induced drifts/warningsAndDrifts
        public int numberOfDriftsInduced;

        public long instancesLog;

        //
        protected int windowObservationSize;
        protected double accuracyWindowLearner;
        protected Random subspaceRandomBaseLearner;

        protected int[] accClassifierArray;
        protected int lastIndex;

        private void init(int indexOriginal, Classifier instantiatedClassifier,
                          BasicClassificationPerformanceEvaluator evaluatorInstantiated,
                          long instancesSeen,  boolean disableDriftDetector,
                          ClassOption driftOption,
                          CopyOnWriteArrayList<Pair<Boolean, Integer>> warningsAndDrifts, int windowObservationSize) {
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;

            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.disableDriftDetector = disableDriftDetector;

            this.warningsAndDrifts = warningsAndDrifts;
            this.windowObservationSize = windowObservationSize;
            this.instancesLog = 0;

            if(!this.disableDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(driftOption)).copy();
            }



            this.numberOfDriftsDetected = this.numberOfDriftsInduced = 0;
        }

        // Create to simulate "Bagging" only, i.e., no random subspaces.
        public SOREClassifier(int indexOriginal, Classifier instantiatedClassifier,
                                                BasicClassificationPerformanceEvaluator evaluatorInstantiated,
                                                long instancesSeen, boolean disableDriftDetector,
                                                ClassOption driftOption,
                                                 CopyOnWriteArrayList<Pair<Boolean, Integer>> warningsAndDrifts, int windowObservationSize) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen,
                    disableDriftDetector, driftOption,
                    warningsAndDrifts, windowObservationSize
            );

            this.featureIndexes = null;
            this.subset = null;
        }

        // Create the subspaces for the current model.
        public SOREClassifier(int indexOriginal, Classifier instantiatedClassifier,
                                                BasicClassificationPerformanceEvaluator evaluatorInstantiated,
                                                long instancesSeen, boolean disableDriftDetector,
                                                ClassOption driftOption,
                                                ArrayList<Integer> featuresIndexes, Instance instance,
                                                CopyOnWriteArrayList<Pair<Boolean, Integer>> warningsAndDrifts, int windowObservationSize) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen,
                    disableDriftDetector, driftOption, warningsAndDrifts, windowObservationSize);

            // Features + class (last index)
            this.featureIndexes = new int[featuresIndexes.size()];
            ArrayList<Attribute> attSub = new ArrayList<Attribute>();

            // Add attributes of the selected subset
            for(int i = 0 ; i < featuresIndexes.size() ; ++i) {
                attSub.add(instance.attribute(featuresIndexes.get(i)));
                this.featureIndexes[i] = featuresIndexes.get(i);
            }

            this.subset = new Instances("Subsets Candidate Instances", attSub, 100);
            this.subset.setClassIndex(this.subset.numAttributes()-1);
            prepareRandomSubspaceInstance(instance,1);
        }

        public void prepareRandomSubspaceInstance(Instance instance, double weight) {
            // If there is any instance lingering in the subset, remove it.
            while(this.subset.numInstances() > 0)
                this.subset.delete(0);

            double[] values = new double[this.subset.numAttributes()];
            for(int j = 0 ; j < this.subset.numAttributes() ; ++j)
                values[j] = instance.value(this.featureIndexes[j]);

            // Set the class value for each value array.
            values[values.length-1] = instance.classValue();
            DenseInstance subInstance = new DenseInstance(1.0, values);
            subInstance.setWeight(weight);
            subInstance.setDataset(this.subset);
            this.subset.add(subInstance);
        }

        private ArrayList<Integer> applySubsetResetStrategy(Instance instance, Random random) {
            if(this.subset != null) {
                ArrayList<Integer> fIndexes = new ArrayList<Integer>();
                for(int j = 0 ; j < instance.numAttributes() ; ++j)
                    fIndexes.add(j);
                // Remove the class label... (it will be added latter)
                fIndexes.remove(instance.classIndex());

                for(int j = 0 ; j < instance.numAttributes() - this.featureIndexes.length ; ++j)
                    fIndexes.remove(random.nextInt(fIndexes.size()));
                // Adding the class label...
                fIndexes.add(instance.classIndex());
                return fIndexes;
            }
            return null;
        }

        public void reset(Instance instance, long instancesSeen, Random random) {
            this.classifier.resetLearning();
            this.evaluator.reset();
            this.createdOn = instancesSeen;
            this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();

            if(this.subset != null) {
                ArrayList<Integer> fIndexes = this.applySubsetResetStrategy(instance, random);
                for(int i = 0 ; i < fIndexes.size() ; ++i)
                    this.featureIndexes[i] = fIndexes.get(i);
                ArrayList<Attribute> attSub = new ArrayList<Attribute>();
                // Add attributes of the selected subset
                for(int i = 0 ; i < this.featureIndexes.length ; ++i)
                    attSub.add(instance.attribute(this.featureIndexes[i]));

                this.subset = new Instances("Subsets Candidate Instances", attSub, 100);
                this.subset.setClassIndex(this.subset.numAttributes()-1);
                prepareRandomSubspaceInstance(instance, 1);
            }

        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen, Random random) {
            boolean correctlyClassifies;
            // The subset object will be null if we are training with all features
            if(this.subset != null) {
                // Selecting just the subset of features that we are going to use
                prepareRandomSubspaceInstance(instance, weight);

                // After prepareRandomSubspaceInstance, index 0 of subset holds the instance with this learner subspaces
                this.classifier.trainOnInstance(this.subset.get(0));
                correctlyClassifies = this.classifier.correctlyClassifies(this.subset.get(0));
            }
            else {
                Instance weightedInstance = instance.copy();
                weightedInstance.setWeight(instance.weight() * weight);
                this.classifier.trainOnInstance(weightedInstance);
                correctlyClassifies = this.classifier.correctlyClassifies(instance);
            }

            if(!this.disableDriftDetector) {

                /*********** drift detection ***********/
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if (this.driftDetectionMethod.getChange()) {
                    this.numberOfDriftsDetected++;
                    // There was a change, this model must be reset
                    //this.reset(instance, instancesSeen, random);
                    //this.drifts.add(this.indexOriginal);
                    this.warningsAndDrifts.add(new Pair<>(false, this.indexOriginal));
                }
            }
            this.updateMatrixConfusion(correctlyClassifies);

        }


        private void updateMatrixConfusion(boolean correctlyClassifies) {
            double acc = 0.0;

            if (this.windowObservationSize <= 0)
                return;

            if (accClassifierArray == null) {
                accClassifierArray = new int[this.windowObservationSize+2];
                lastIndex = -1;
            }

            if (lastIndex == this.windowObservationSize-1) {
                lastIndex = -1;

            }

            if (accClassifierArray[accClassifierArray.length-2] < this.windowObservationSize) {
                accClassifierArray[accClassifierArray.length-2] ++;
            }

            lastIndex++;
            accClassifierArray[accClassifierArray.length-1] -=
                    accClassifierArray[lastIndex];
            accClassifierArray[lastIndex] = correctlyClassifies ? 1 : 0;


            accClassifierArray[accClassifierArray.length-1] +=
                    accClassifierArray[lastIndex];

            acc = (double) accClassifierArray[accClassifierArray.length-1]/accClassifierArray[accClassifierArray.length-2];
            accuracyWindowLearner = acc;

        }


        /**
         * @param instance
         * @return votes for the given instance
         */
        public double[] getVotesForInstance(Instance instance) {
            if(this.subset != null) {
                prepareRandomSubspaceInstance(instance, 1);
                // subset.get(0) returns the instance transformed to the correct subspace (i.e. current model subspace).
                DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(this.subset.get(0)));

                return vote.getArrayRef();
            }
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
    }

    /***
     * Inner class to assist with the multi-thread execution.
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private SOREClassifier learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;
        final private Random random;

        public TrainingRunnable(SOREClassifier learner, Instance instance,
                                double weight, long instancesSeen, Random random) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
            this.random = random;
        }

        @Override
        public void run() {
            learner.trainOnInstance(instance, weight, this.instancesSeen, this.random);
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }

    }

    public class Pair<A, B> {
        public final A first;
        public final B second;

        public Pair(A first, B second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public String toString() {
            return " bkg: " + first + " idx " + second;
        }
    }
}