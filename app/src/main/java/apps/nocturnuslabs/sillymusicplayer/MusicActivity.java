package apps.nocturnuslabs.sillymusicplayer;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

public class MusicActivity extends AppCompatActivity {
    public Boolean mood;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_music);

        mood = getIntent().getBooleanExtra("Mood",true);

    }
}
