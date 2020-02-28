/**
 * This include code translated from the JAMA package. The code is released
 * into the public domain, but contains the following license information.
 * 
 * 
 * Copyright Notice 
 * 
 * This software is a cooperative product of The MathWorks 
 * and the National Institute of Standards and Technology (NIST) which 
 * has been released to the public domain. Neither The MathWorks nor 
 * NIST assumes any responsibility whatsoever for its use by other parties, 
 * and makes no guarantees, expressed or implied, about its quality, reliability, 
 * or any other characteristic.
 */
package optim4j.testbeds;

import utils.RealMath;

/**
 * QR decomposition of an arbitrary real valued matrix.
 */
public final class QR {

	public static final double[][][] qr(final double[][] mat) {

		// copy original matrix
		final int m = mat.length;
		final int n = mat[0].length;
		final double[][] QR = new double[m][];
		for (int i = 0; i < m; ++i) {
			QR[i] = mat[i].clone();
		}

		// apply Householder transformations
		final double[] Rdiag = new double[n];
		for (int k = 0; k < n; k++) {
			double nrm = 0;
			for (int i = k; i < m; i++) {
				nrm = RealMath.hypot(nrm, QR[i][k]);
			}
			if (nrm != 0.0) {
				if (QR[k][k] < 0) {
					nrm = -nrm;
				}
				for (int i = k; i < m; i++) {
					QR[i][k] /= nrm;
				}
				QR[k][k] += 1.0;
				for (int j = k + 1; j < n; j++) {
					double s = 0.0;
					for (int i = k; i < m; i++) {
						s += QR[i][k] * QR[i][j];
					}
					s = -s / QR[k][k];
					for (int i = k; i < m; i++) {
						QR[i][j] += s * QR[i][k];
					}
				}
			}
			Rdiag[k] = -nrm;
		}

		// compute the matrix R
		final double[][] R = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i < j) {
					R[i][j] = QR[i][j];
				} else if (i == j) {
					R[i][j] = Rdiag[i];
				} else {
					R[i][j] = 0.0;
				}
			}
		}

		// compute the matrix Q
		final double[][] Q = new double[m][n];
		for (int k = n - 1; k >= 0; k--) {
			for (int i = 0; i < m; i++) {
				Q[i][k] = 0.0;
			}
			Q[k][k] = 1.0;
			for (int j = k; j < n; j++) {
				if (QR[k][k] != 0) {
					double s = 0.0;
					for (int i = k; i < m; i++) {
						s += QR[i][k] * Q[i][j];
					}
					s = -s / QR[k][k];
					for (int i = k; i < m; i++) {
						Q[i][j] += s * QR[i][k];
					}
				}
			}
		}
		return new double[][][] { Q, R };
	}
}
