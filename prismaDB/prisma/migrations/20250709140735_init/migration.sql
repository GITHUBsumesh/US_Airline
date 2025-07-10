-- CreateTable
CREATE TABLE "US_Airline" (
    "id" SERIAL NOT NULL,
    "year" INTEGER NOT NULL,
    "month" INTEGER NOT NULL,
    "carrier" TEXT NOT NULL,
    "carrier_name" TEXT NOT NULL,
    "airport" TEXT NOT NULL,
    "airport_name" TEXT NOT NULL,
    "arr_flights" DOUBLE PRECISION NOT NULL,
    "arr_del15" DOUBLE PRECISION NOT NULL,
    "carrier_ct" DOUBLE PRECISION NOT NULL,
    "weather_ct" DOUBLE PRECISION NOT NULL,
    "nas_ct" DOUBLE PRECISION NOT NULL,
    "security_ct" DOUBLE PRECISION NOT NULL,
    "late_aircraft_ct" DOUBLE PRECISION NOT NULL,
    "arr_cancelled" DOUBLE PRECISION NOT NULL,
    "arr_diverted" DOUBLE PRECISION NOT NULL,
    "arr_delay" DOUBLE PRECISION NOT NULL,
    "carrier_delay" DOUBLE PRECISION NOT NULL,
    "weather_delay" DOUBLE PRECISION NOT NULL,
    "nas_delay" DOUBLE PRECISION NOT NULL,
    "security_delay" DOUBLE PRECISION NOT NULL,
    "late_aircraft_delay" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "US_Airline_pkey" PRIMARY KEY ("id")
);
