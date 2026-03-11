package moa.classifiers.AutoML.RandomSearch;

import java.io.Serializable;

public abstract class Parameter implements Serializable {
	
	public int type;
	public String parameter;
	public boolean direction;
	public void changeParameter() {
	}
	public void getValue() {
		
	}
	
	
}
