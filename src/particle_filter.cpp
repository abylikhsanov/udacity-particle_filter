/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include<random>
#include<map>
#include "helper_functions.h"

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 105;
	default_random_engine gen;
	normal_distribution<double> x_std(x,std[0]);
	normal_distribution<double> y_std(y,std[1]);
	normal_distribution<double> theta_std(theta,std[2]);
	for(unsigned int i = 0; i<num_particles;++i){

		double sample_x; double sample_y; double sample_theta;
		sample_x = x_std(gen);
		sample_y = y_std(gen);
		sample_theta = theta_std(gen);
		Particle p = Particle();
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1.0;
		particles.emplace_back(p);
        weights.emplace_back(1);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {


	std::normal_distribution<double> x_std(0,std_pos[0]);
	std::normal_distribution<double> y_std(0,std_pos[1]);
	std::normal_distribution<double> theta_std(0,std_pos[2]);

    for(auto& elem : particles){

        if(fabs(yaw_rate)<0.001){
            elem.x += (velocity) * delta_t * cos(elem.theta) + x_std(gen);
            elem.y += (velocity) * delta_t * sin(elem.theta) + y_std(gen);
            elem.theta += theta_std(gen);
        }

        else{
            elem.x += (velocity / yaw_rate) * (sin(elem.theta + yaw_rate * delta_t) - sin(elem.theta)) + x_std(gen);
            elem.y += (velocity / yaw_rate) * (cos(elem.theta) - cos(elem.theta + yaw_rate * delta_t)) + y_std(gen);
            elem.theta += yaw_rate * delta_t + theta_std(gen);
        }
    }



}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    double denom1 = 2*std_landmark[0]*std_landmark[0];
    double denom2 = 2*std_landmark[1]*std_landmark[1];
    double denom3 = 2*M_PI*std_landmark[0]*std_landmark[1];
    unsigned int count = 0;
	for(auto& elem : particles){ // Looping for every particle

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;


        vector<LandmarkObs> converted_obs = observations; // Copying the constant observation object

        for(unsigned int i= 0; i<observations.size(); ++i) {
            converted_obs.at(i).id = 0;
            // Simply converting the coordinates and updating the X and Y of the copied observation object
            converted_obs.at(i).x = (elem.x + (observations.at(i).x * cos(elem.theta)) -
                                     (observations.at(i).y * sin(elem.theta)));
            converted_obs.at(i).y = (elem.y + (observations.at(i).x * sin(elem.theta)) +
                                     (observations.at(i).y * cos(elem.theta)));
        }

        elem.weight = 1.0;
        for(auto& obs : converted_obs){ // For each transformed observation
            double min_distance = sensor_range; //
            obs.id = 0;
            // for each landmark
            for(int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
                double landmark_x = map_landmarks.landmark_list[j].x_f;
                double landmark_y = map_landmarks.landmark_list[j].y_f;

                //calculate the distance from the transformed observation to the landmark
                //double calc_dist =sqrt(pow(trans_observations[i].x-landmark_x, 2.0)+pow(trans_observations[i].y-landmark_y,2.0));
                //double calc_dist = dist(landmark_x, landmark_y, trans_observations[i].x, trans_observations[i].y);
                double calc_dist = dist(obs.x, obs.y, landmark_x, landmark_y);
                //assign landmark to the observation with the shortest distance
                if(calc_dist < min_distance) {
                    min_distance = calc_dist;
                    obs.id = j;
                }
            }

            if(obs.id != 0){
                double meas_x = obs.x;
                double meas_y = obs.y;
                float mu_x = map_landmarks.landmark_list.at(obs.id).x_f;
                float mu_y = map_landmarks.landmark_list.at(obs.id).y_f;
                // Calculate normalization term:
                long double multipler = 1/denom3*exp(-(pow(meas_x-mu_x,2.0)/denom1+pow(meas_y-mu_y,2.0)/denom2));
                if(multipler > 0) {
                    elem.weight *= multipler;
                }

            }

            associations.push_back(obs.id+1);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);

		}
        elem = SetAssociations(elem,associations,sense_x,sense_y);
        weights[count] = elem.weight;
        ++count;
	}

}

void ParticleFilter::resample() {

    default_random_engine gen;
    discrete_distribution<unsigned int> distribution(weights.begin(),weights.end());

    vector<Particle> newParticles;

    for(unsigned int i = 0; i<num_particles; ++i){
        unsigned int id = distribution(gen);
        newParticles.emplace_back(particles.at(id));
    }

    particles = newParticles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
