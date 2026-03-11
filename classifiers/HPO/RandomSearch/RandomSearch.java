package moa.classifiers.AutoML.RandomSearch;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.classifiers.trees.ARTEHoeffdingTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;

import java.io.File;
import java.io.FileWriter;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class RandomSearch extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler, Serializable {

	public FileOption fileOption = new FileOption("ConfigurationFile", 'f', "Configuration file in json format.",
            "HT.json", ".json", false);

    public FileOption writeFile = new FileOption("writeFile", 'a', "Write to file.",
            "elec_rs.csv", ".csv", false);

    public IntOption stateGrace = new IntOption("stateGrace", 'g',
            "Periodic evaluations of the state back.", 1000, 1, Integer.MAX_VALUE);

    public IntOption rsO = new IntOption("rsO", 's',
            "Random Seed Value.", 1, 1, Integer.MAX_VALUE);

    public FlagOption writeToFile = new FlagOption("writeToFile", 'w',
            "write solutions to a file");

    public FlagOption randomInitParam = new FlagOption("randomInitParam", 'R',
            "random initial parameter");

    public FlagOption disablevoteBackOption = new FlagOption("disablevoteBackOption", 'V',
            "voteBackOption");




    double mean_difference = 0.0;

    int last_comp = 0;

    int new_comp = 0;


    public long instances;

    public int count_better = 0;

    public int count_worst = 0;

    public long statesEvaluated;

    public boolean voteBack = false;

    String learner;

    public Classifier classifier;

    public Classifier stateforward;

    ArrayList<Parameter> classifierParameters;

    ArrayList<Parameter> stateParameters;

    //ArrayList<ArrayList<Parameter>> configs;

    int stateSelected;

    int numericalHyperparams;

    int[][] possibleStates;

    double[] probStateSelection;

    protected BasicClassificationPerformanceEvaluator evaluatorClassifier;

    protected BasicClassificationPerformanceEvaluator evaluatorstateforward;

    boolean checkAccuracy;

    int evaluation_instances ;

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        //Main classifiers votes
        if(this.voteBack){
            return this.stateforward.getVotesForInstance(inst);
        }else{
            return this.classifier.getVotesForInstance(inst);
        }

    }

    Random rejection;

    @Override
    public void resetLearningImpl() {
        //this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        //System.out.println("HEY");

        this.classifierParameters = new ArrayList<Parameter>();
        this.stateParameters = new ArrayList<Parameter>();
        this.evaluatorClassifier = new BasicClassificationPerformanceEvaluator();
        this.evaluatorstateforward = new BasicClassificationPerformanceEvaluator();

        this.evaluatorClassifier.precisionRecallOutputOption.setValue(true);
        this.evaluatorstateforward.precisionRecallOutputOption.setValue(true);

        this.setConfigurations();
        this.instances = 0;
        this.statesEvaluated = 0;
        this.rejection = new Random(this.rsO.getValue());
        this.evaluation_instances = 0;
    }



    public void setConfigurations(){
        //Set all hyperparameters and store them for future control
        try {
            this.numericalHyperparams = 0;
            //Jackson object
            ObjectMapper objectMapper = new ObjectMapper();

            // Read JSON file
            JsonNode rootNode = objectMapper.readTree(new File(fileOption.getValue()));

            this.learner = rootNode.get("algorithm").asText();

            //create classifier and stateforward (for hyper-parameter testing)
            this.classifier = (Classifier) ClassOption.cliStringToObject(rootNode.get("algorithm").asText(), Classifier.class, null);
            this.classifier.resetLearning();

            this.stateforward = (Classifier) ClassOption.cliStringToObject(rootNode.get("algorithm").asText(), Classifier.class, null);
            this.stateforward.resetLearning();
            //Set initial parameters
            for (JsonNode paramNode : rootNode.get("parameters")) {
                String parameterName = paramNode.get("parameter").asText();
                //set parameter values
                Field field = this.classifier.getClass().getField( parameterName );
                String type = paramNode.get("type").asText();
                //filter by type
                switch(type) {
                    case "categorical":
                        JsonNode values = paramNode.get("values");
                        //store each categorical value
                        String[] Values = new String[values.size()];
                        for (int i = 0; i < values.size(); i++) {
                            Values[i] = values.get(i).asText();
                        }
                        int active;
                        if(!this.randomInitParam.isSet()){
                            active = paramNode.get("active").asInt();
                        }else{
                            active = this.classifierRandom.nextInt(Values.length);
                        }

                        //set random first value

                        field.set(this.classifier, Values[active]);

                        this.classifierParameters.add(new CategoricalParameter(parameterName, Values, active, new Random(this.rsO.getValue())));
                        this.stateParameters.add(new CategoricalParameter(parameterName, Values, active, new Random(this.rsO.getValue())));
                        break;

                    case "integer":
                        //Add integer hyperparameter
                        //int value = paramNode.get("value").asInt();
                        //field.set(this.classifier, value);
                        JsonNode range = paramNode.get("range");
                        int[] Range = new int[range.size()];
                        for (int i = 0; i < range.size(); i++) {
                            Range[i] = range.get(i).asInt();
                        }
                        int value;
                        if(!this.randomInitParam.isSet()) {
                            value = paramNode.get("value").asInt();
                        }
                        else{
                            value = this.classifierRandom.nextInt(Range[1] + 1 - Range[0]) + Range[0];
                        }
                        field.set(this.classifier, value);

                        this.classifierParameters.add(new IntParameter(parameterName, value, Range, new Random(this.rsO.getValue())));
                        this.stateParameters.add(new IntParameter(parameterName, value, Range, new Random(this.rsO.getValue())));
                        this.numericalHyperparams++;
                        break;

                    case "double":

                        //Add double hyperparameter
                        //double value_d = paramNode.get("value").asDouble();
                        //field.set(this.classifier, value_d);

                        JsonNode range_d = paramNode.get("range");
                        double[] Range_d = new double[range_d.size()];
                        for (int i = 0; i < range_d.size(); i++) {
                            Range_d[i] = range_d.get(i).asDouble();
                        }


                        double value_d;
                        if(!this.randomInitParam.isSet()) {
                            value_d = paramNode.get("value").asDouble();
                        }
                        else{
                            value_d = Range_d[0] + (Range_d[1] - Range_d[0]) *  this.classifierRandom.nextDouble();
                        }

                        field.set(this.classifier, value_d);

                        this.classifierParameters.add(new DoubleParameter(parameterName, value_d, Range_d, new Random(this.rsO.getValue())));
                        this.stateParameters.add(new DoubleParameter(parameterName, value_d, Range_d, new Random(this.rsO.getValue())));

                        this.numericalHyperparams++;
                        break;
                }
            }
        }catch(Exception e) {
            e.printStackTrace(System.out);
        }
    }

    public void deepCopyList() {
        //Swap parameters from Stateforward to main classifier
        for (int i = 0; i < this.classifierParameters.size(); i++) {
            Parameter parameterS = this.stateParameters.get(i);
            Parameter parameterC = this.classifierParameters.get(i);
            switch(parameterS.type) {
                case 0:
                    ((IntParameter) parameterC).value = ((IntParameter) parameterS).value;
                    ((IntParameter) parameterC).range = ((IntParameter) parameterS).range;
                    break;
                case 1:
                    ((DoubleParameter) parameterC).value = ((DoubleParameter) parameterS).value;
                    ((DoubleParameter) parameterC).range = ((DoubleParameter) parameterS).range;
                    break;
                case 2:
                    ((CategoricalParameter) parameterC).values = ((CategoricalParameter) parameterS).values;
                    ((CategoricalParameter) parameterC).active = ((CategoricalParameter) parameterS).active;
                    break;
            }

        }
    }





    public void checkParameterChange(){
        try {
            double stateAcc = 0;
            double classifierAcc = 0;

            if (this.statesEvaluated != 0) {
                stateAcc = this.evaluatorstateforward.getPerformanceMeasurements()[1].getValue();
                classifierAcc = this.evaluatorClassifier.getPerformanceMeasurements()[1].getValue();
            }





            // Swaping classifiers if stateforward acc is greater than main classifier
            if(stateAcc > classifierAcc) {
                //System.out.println("ne pas possible");
                this.swapClassifiers();
            }
            else{
                this.stateforward = this.classifier.copy();
                this.stateforward.resetLearning();
            }


            //Parameter change of stateforward
            for (Parameter parameter : this.stateParameters) {
                this.changeStateParameter(parameter);
            }

            //reset evaluators
            this.evaluatorClassifier.reset();
            this.evaluatorstateforward.reset();

        }catch(Exception e) {
            e.printStackTrace(System.out);
        }
    }

    public void swapClassifiers() {
        this.classifier = this.stateforward.copy();
        //this.stateforward.resetLearning();
        this.stateforward.resetLearning();
        this.deepCopyList();
        //this.classifierParameters = (ArrayList<Parameter>) this.stateParameters.clone();
    }

    public void changeStateParameter(Parameter parameter) throws NoSuchFieldException, IllegalAccessException {
        //Change the parameter in the classifier (stateforward)
        parameter.changeParameter();
        Field field = this.stateforward.getClass().getField( parameter.parameter );
        switch(parameter.type) {
            case 0:
                field.set(this.stateforward, ((IntParameter) parameter).value );
                break;
            case 1:
                field.set(this.stateforward, ((DoubleParameter) parameter).value );
                break;
            case 2:
                CategoricalParameter p = ((CategoricalParameter) parameter);
                field.set(this.stateforward, p.values[p.active]);
        }
    }



    @Override
    public void trainOnInstanceImpl(Instance inst) {
        //check for change in parameters
        this.evaluation_instances++;
        double stateAcc = 0;
        double classifierAcc = 0;
        try {
            stateAcc = this.evaluatorstateforward.getPerformanceMeasurements()[1].getValue();
            classifierAcc = this.evaluatorClassifier.getPerformanceMeasurements()[1].getValue();
        }catch(Exception e){

        }



        if ( (this.statesEvaluated == 0) || this.evaluation_instances >= this.stateGrace.getValue() ) {
            //System.out.println(this.evaluation_instances);
            this.checkAccuracy = false;
            //System.out.println("State Acc: " + stateAcc + " Classifier Acc: " + classifierAcc);
            this.checkParameterChange();
            this.statesEvaluated++;
            this.evaluation_instances = 0;
        }

//        double stateAcc = 0;
//        double classifierAcc = 0;
//
//        stateAcc = this.evaluatorstateforward.getPerformanceMeasurements()[1].getValue();
//        classifierAcc = this.evaluatorClassifier.getPerformanceMeasurements()[1].getValue();
//
        int instances_eval = Math.toIntExact(this.instances % 1000);




//        this.voteBack = (!this.disablevoteBackOption.isSet()) && (stateAcc - classifierAcc) / 100 >  Math.sqrt( (Math.log(1.0 / 0.1))
//                / ( 2.0 *  instances_eval));

        this.voteBack = (!this.disablevoteBackOption.isSet()) && (stateAcc > classifierAcc)  ;


        Instance unweightedInst = (Instance) inst.copy();
        unweightedInst.setWeight(1.0);

        //update evaluators (update metrics)
        InstanceExample example = new InstanceExample(unweightedInst);
        this.evaluatorClassifier.addResult(example, this.classifier.getVotesForInstance(inst));
        this.evaluatorstateforward.addResult(example, this.stateforward.getVotesForInstance(inst));

        //train classifiers
        this.classifier.trainOnInstance(inst);
        this.stateforward.trainOnInstance(inst);
        this.instances++;
    }
    
    @Override
    public long measureByteSize() {
        return SizeOf.sizeOf(this) + this.classifier.measureByteSize() + this.stateforward.measureByteSize();
    }
    

    @Override
    protected Measurement[] getModelMeasurementsImpl() {

        ArrayList<Measurement> parameters = new ArrayList<>();

        for (Measurement m : this.classifier.getModelMeasurements()) {
            parameters.add(m);
        }
        for(Parameter p : this.classifierParameters){
            switch (p.type){
                case 0:
                    parameters.add(new Measurement(p.parameter, ((IntParameter) p).value));
                    break;
                case 1:
                    parameters.add(new Measurement(p.parameter, ((DoubleParameter) p).value));
                    break;
                case 2:
                    parameters.add(new Measurement(p.parameter, ((CategoricalParameter) p).active)) ;
                    break;
            }
        }
        Measurement[] measurements = new Measurement[parameters.size()];
        measurements = parameters.toArray(measurements);

        return measurements;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }


}
