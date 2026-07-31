package util;

import org.eclipse.mosaic.fed.application.ambassador.simulation.perception.PerceptionModuleOwner;
import org.eclipse.mosaic.fed.application.ambassador.simulation.perception.errormodels.PerceptionModifier;
import org.eclipse.mosaic.fed.application.ambassador.simulation.perception.index.objects.SpatialObject;
import org.eclipse.mosaic.fed.application.ambassador.simulation.perception.index.objects.VehicleObject;
import org.eclipse.mosaic.lib.math.RandomNumberGenerator;

import java.util.List;

/**
 * Applies independent Gaussian speed error to perceived vehicles.
 */
public final class SpeedModifier implements PerceptionModifier {

    private final RandomNumberGenerator random;
    private final double standardDeviation;

    public SpeedModifier(RandomNumberGenerator random, double standardDeviation) {
        if (standardDeviation < 0.0) {
            throw new IllegalArgumentException("Speed standard deviation must be nonnegative");
        }
        this.random = random;
        this.standardDeviation = standardDeviation;
    }

    @Override
    public <T extends SpatialObject<?>> List<T> apply(
            PerceptionModuleOwner owner,
            List<T> perceivedObjects
    ) {
        for (T perceivedObject : perceivedObjects) {
            if (perceivedObject instanceof VehicleObject vehicle) {
                double noisySpeed = random.nextGaussian(vehicle.getSpeed(), standardDeviation);
                vehicle.setSpeed(Math.max(0.0, noisySpeed));
            }
        }
        return perceivedObjects;
    }
}
