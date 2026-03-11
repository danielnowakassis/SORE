package moa.classifiers.AutoML.RandomSearch;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class IntParameter extends Parameter implements Serializable {
	
	
	public int value;
	public int[] range;
	public Random random;

	public IntParameter(String parameter, int value, int[] range, Random random) {
		this.parameter = parameter;
		this.value = value;
		this.range = range;
		this.type = 0;
		this.random = random;
	}

	@Override
	public void changeParameter() {
		this.value = this.random.nextInt((this.range[1] - this.range[0] + 1)) + this.range[0];
		this.value = Math.min(this.range[1], Math.max(this.range[0], this.value));
	}

	@Override
	public String toString() {
		return "IntParameter [name=" + this.parameter + ", value=" + value + ", range=" + Arrays.toString(range) + "]";
	}
	
	
	
	
	 
	
}
