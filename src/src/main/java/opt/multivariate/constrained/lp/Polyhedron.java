package opt.multivariate.constrained.lp;

/**
 *
 * @author michael
 */
public class Polyhedron {

	private final double[][] myA;
	private final double[] myB;
	public final int myD, myNumLe, myNumGe;

	/**
	 *
	 * @param amat
	 * @param bvec
	 * @param numlesseq
	 * @param numgreeq
	 */
	public Polyhedron(final double[][] amat, final double[] bvec, final int numlesseq, final int numgreeq) {
		myA = amat;
		myB = bvec;
		myNumLe = numlesseq;
		myNumGe = numgreeq;
		myD = amat[0].length;
	}

	/**
	 *
	 * @return
	 */
	public final double[][] getA() {
		return myA;
	}

	/**
	 *
	 * @return
	 */
	public final double[] getB() {
		return myB;
	}
}
