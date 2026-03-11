package moa.classifiers.AutoML.RandomSearch;

import java.io.Serializable;
import java.util.Random;

public class CategoricalParameter extends Parameter implements Serializable {
	
	public String[] values;
	public int active;
	public Random random;

	public CategoricalParameter(String parameter,String[] values, int active, Random random) {
		this.parameter = parameter;
		this.values = values;
		this.active = active;
		this.type = 2;
		this.random = random;
	}

	@Override
	public void changeParameter() {
		this.active = this.random.nextInt(this.values.length);
	}
	
	@Override
	public String toString() {
		return "CategoricalParameter [active=" + values[active] + "]";
	}
	
	
	
}
