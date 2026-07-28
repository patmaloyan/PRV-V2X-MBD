package util;

import org.eclipse.mosaic.lib.geo.CartesianPoint;
import org.eclipse.mosaic.lib.geo.MutableCartesianPoint;

public class SensorErrorModel {

    public Pair<MutableCartesianPoint, MutableCartesianPoint> addPositionNoise(CartesianPoint truePosition) {
        return new Pair<>(
                new MutableCartesianPoint(truePosition.getX(), truePosition.getY(), truePosition.getZ()),
                new MutableCartesianPoint(0, 0, 0)
        );
    }

    public Pair<Double, Double> addSpeedNoise(double trueVelocity) {
        return new Pair<>(trueVelocity, 0.0);
    }

    public Pair<Double, Double> addAccelerationNoise(double acceleration, long currentTime) {
        return new Pair<>(acceleration, 0.0);
    }

    public Pair<Double, Double> addHeadingNoise(double trueHeading, double velocity) {
        return new Pair<>(trueHeading, 0.0);
    }
}
