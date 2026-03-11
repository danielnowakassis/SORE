package moa.classifiers.AutoML.RandomSearch;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class DoubleParameter extends Parameter implements Serializable {


	public double value;
	public double[] range;
	public Random random;

	public DoubleParameter(String parameter, double value, double[] range, Random random) {
		this.parameter = parameter;
		this.value = value;
		this.range = range;
		this.type = 1;
		this.random = random;
	}
	
	@Override
	public void changeParameter() {
		this.value = this.range[0] + ( (this.range[1] - this.range[0]) * this.random.nextDouble() ) ;
		this.value = Math.min(this.range[1], Math.max(this.range[0], this.value));
	}
	
	@Override
	public String toString() {
		return "DoubleParameter [name=" + this.parameter +", value=" + value + ", range=" + Arrays.toString(range) + "]";
	}
}
