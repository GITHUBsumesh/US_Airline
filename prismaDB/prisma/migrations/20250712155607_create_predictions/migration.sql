-- CreateTable
CREATE TABLE "Prediction" (
    "id" SERIAL NOT NULL,
    "predictedValue" INTEGER NOT NULL,
    "inputFeatures" JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Prediction_pkey" PRIMARY KEY ("id")
);
