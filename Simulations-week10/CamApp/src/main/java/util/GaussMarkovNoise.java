package util;

import org.eclipse.mosaic.lib.math.RandomNumberGenerator;

public class GaussMarkovNoise {
    private final double alpha;
    private final double sigma;
    private double currentValue;
    private final RandomNumberGenerator random;

    /**
     * @param alpha        Korrelation (typischerweise alpha = exp(-dt/τ))
     * @param sigma        Standardabweichung des Rauschens
     * @param initialValue Startwert (z.B. 0.0)
     */
    public GaussMarkovNoise(double alpha, double sigma, double initialValue, RandomNumberGenerator random) {
        this.alpha = alpha;
        this.sigma = sigma;
        this.currentValue = initialValue;
        this.random = random;
    }

    /**
     * Berechnet den nächsten Wert im Gauss-Markov-Prozess.
     *
     * @return Neuer Prozesswert.
     */
    public double nextValue() {
        double noise = Math.abs(random.nextGaussian()) * Math.sqrt(1 - alpha * alpha) * sigma;
        currentValue = alpha * currentValue + noise;
        return currentValue;
    }

}
